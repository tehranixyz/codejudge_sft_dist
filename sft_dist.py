import os
os.environ["HF_HUB_OFFLINE"]='1'
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_CACHE_DIR"] = "/home/codejudge/sft/.wandbcache"
os.environ['HF_HOME'] = '/home/codejudge/sft/.hfcache'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from typing import Callable, Optional
import warnings
import wandb
import json
import os.path
import sys
sys.path.append("..")
import argparse
import numpy as np
import traceback
import datasets
from transformers import AutoTokenizer
import pathlib
import logging
import random
import torch
from functools import partial
import transformers
from datasets import load_from_disk, load_dataset, concatenate_datasets
from rich.logging import RichHandler
from transformers import set_seed
from trl import SFTTrainer, SFTConfig
# from unsloth import FastLanguageModel
from transformers import TrainingArguments, AutoModelForCausalLM
import logging
from rich.logging import RichHandler
import pandas as pd
import time
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from accelerate import PartialState
from accelerate import Accelerator
from trl.extras.dataset_formatting import get_formatting_func_from_dataset

LANGUAGE_CONVENTIONS={
    'cpp':'C++',
    'java':'Java',
    'python':'Python'
}

def apply_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# CodeJudge Target Models
# google/gemma-2-27b-it
# mistralai/Codestral-22B-v0.1

def parse_args():
    parser = argparse.ArgumentParser(description="Script for SFT Tuning.")

    # Required parameters
    parser.add_argument('--train_dataset_name', type=str, required=True, help="Name of the training dataset.")
    parser.add_argument('--validation_dataset_name', type=str, required=True, help="Name of validation dataset.")
    parser.add_argument('--dataset_loc', type=str, required=True, help="Location of the dataset.")
    parser.add_argument('--llm_path', type=str, required=True, help="Location of LLM model")
    parser.add_argument('--output_loc', type=str, required=True, help="Location of the output.")
    parser.add_argument('--log_dir', type=str, required=True, help="Location of the log.")
    parser.add_argument('--run_name', type=str, required=True, help="Run name for experiment.")
    parser.add_argument('--num_proc_dataset', type=int, required=True, help="Number of processor for dataset map.")
    parser.add_argument('--dataset_num_proc', type=int, required=True, help="Number of processor to use to tokenize the data.")
    # Default training parameters
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", help="Name of the attention implementation library.") # also eager for gemma
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help="Per device train batch size.")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help="Per device eval batch size.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio")
    parser.add_argument('--optim', type=str, default="adamw_8bit", help="Define optimizer.")
    parser.add_argument('--num_train_epochs', type=float, default=5, help="num of training epochs")
    parser.add_argument('--bias', type=str, default='none', help="bias")
    parser.add_argument('--lora_dropout', type=float, default=0, help="lora_dropout")
    parser.add_argument('--lora_alpha', type=int, default=128, help="lora_alpha")
    parser.add_argument('--lora_rank', type=int, default=64, help="lora rank")
    parser.add_argument('--random_seed', type=int, default=3407, help="random seed")
    parser.add_argument('--random_state', type=int, default=3407, help="random state")
    parser.add_argument('--is_peft', type=bool,default=True, help="Check if the fine-tuning use peft.")
    parser.add_argument('--do_eval', type=bool, default=True, help="Check if users want to do evaluation.")
    parser.add_argument('--max_seq_length', type=int, default=8192, help="max sequence length")
    parser.add_argument('--load_in_4bit', type=bool, default=True, help="Load in 4bit mode?")
    parser.add_argument('--use_nested_quant', type=bool, default=False, help="whether to use nested quant")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default="nf4", help="quantization type for int4")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default="float16", help="int4 compute type")
    parser.add_argument('--logging_steps', type=int, default=1, help="Logging steps.")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="learning rate.")
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help="learning rate schedule")
    parser.add_argument('--dataset_batch_size', type=int, default=1000, help="Number of examples to tokenize")
    parser.add_argument('--save_total_limit', type=int, default=10, help="Maximum number of models to save")
    parser.add_argument('--save_steps', type=int, default=50, help="How often to save the model")
    parser.add_argument('--eval_steps', type=int, default=50, help="How frequent to perform eval")
    parser.add_argument('--split_model', type=bool, default=True, help="split the model across devices")
    parser.add_argument('--use_custom_loss', type=bool, default=False, help="Use SFTTrainer with custom loss")
    parser.add_argument('--lora_target_modules', nargs='+', type=str, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help='A list of modules to apply lora')
    parser.add_argument('--num_train_samples_per_translation', type=int, default=0, help="How many train samples per each translation direction")
    parser.add_argument('--tokenized', type=bool, default=False, help="Loading tokenized dataset")
    args = parser.parse_args()
    return args

def conversational_format(example):
    instruction = f"""You are an expert in code translation between {LANGUAGE_CONVENTIONS[example['source_language']]} and {LANGUAGE_CONVENTIONS[example['target_language']]}.
    Below is the source code written in {LANGUAGE_CONVENTIONS[example['source_language']]}:

    ```
    {example['source_code']}
    ```

    Your task is to translate this {LANGUAGE_CONVENTIONS[example['source_language']]} code into {LANGUAGE_CONVENTIONS[example['target_language']]}.
    Return only the translated {LANGUAGE_CONVENTIONS[example['target_language']]} code, and include the commend |End-of-Code| at the end.
    """

    response = f"```\n{example['target_code']}\n```\n|End-of-Code|"
    # messages = [
    #         {'content': instruction, 'role': 'user'},
    #         {'content': response, 'role': 'assistant'}
    # ]

    return {'prompt': instruction, 'completion': response}

def _prepare_non_packed_dataloader(
    args,
    processing_class,
    dataset,
    dataset_text_field: str,
    max_seq_length,
    formatting_func: Optional[Callable] = None,
    add_special_tokens=True,
    remove_unused_columns=True,
):
    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        outputs = processing_class(
            formatting_func(element),
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        if formatting_func is not None and not isinstance(formatting_func(element), list):
            raise ValueError(
                "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
            )

        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    signature_columns = ["input_ids", "labels", "attention_mask"]

    if dataset.column_names is not None:  # None for IterableDataset
        extra_columns = list(set(dataset.column_names) - set(signature_columns))
    else:
        extra_columns = []

    if not remove_unused_columns and len(extra_columns) > 0:
        warnings.warn(
            "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
            "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
            f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
            "default collator and create your own data collator in order to inspect the unused dataset columns.",
            UserWarning,
        )

    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names if remove_unused_columns else None,
        "batch_size": args.dataset_batch_size,
    }
    if isinstance(dataset, datasets.Dataset):
        map_kwargs["num_proc"] = args.dataset_num_proc  # this arg is not available for IterableDataset
    tokenized_dataset = dataset.map(tokenize, **map_kwargs)

    return tokenized_dataset


def sft_load_dataset(args, logger, tokenizer):

    if args.tokenized:
        logger.info(f"Loading tokenized dataset.")
        dataset_train = load_from_disk(os.path.join(args.dataset_loc, args.train_dataset_name))
        dataset_val = load_from_disk(os.path.join(args.dataset_loc, args.validation_dataset_name))
        return dataset_train, dataset_val

    datafiles = {
        'train': os.path.join(args.dataset_loc, args.train_dataset_name),
        'validation': os.path.join(args.dataset_loc, args.validation_dataset_name)
    }
    dataset = datasets.load_dataset("json", data_files=datafiles)


    # Performing sampling
    if args.num_train_samples_per_translation > 0:
        # Dataset sampling
        logger.info(f"Sampling {args.num_train_samples_per_translation} per each translation direction.")
        ## Filtering
        dataset_dir_1 = (
            dataset["train"]
            .filter(
                lambda example: example["source_language"] == "cpp"
                and example["target_language"] == "python",
                num_proc=args.num_proc_dataset,
            )
            .select(range(args.num_train_samples_per_translation))
        )

        dataset_dir_2 = (
            dataset["train"]
            .filter(
                lambda example: example["source_language"] == "java"
                and example["target_language"] == "cpp",
                num_proc=args.num_proc_dataset,
            )
            .select(range(args.num_train_samples_per_translation))
        )

        dataset_dir_3 = (
            dataset["train"]
            .filter(
                lambda example: example["source_language"] == "java"
                and example["target_language"] == "python",
                num_proc=args.num_proc_dataset,
            )
            .select(range(args.num_train_samples_per_translation))
        )
        # Concatenating
        dataset['train'] = concatenate_datasets([dataset_dir_1, dataset_dir_2, dataset_dir_3])

    dataset = dataset.map(
        conversational_format,
        num_proc=args.num_proc_dataset,
        desc="Conversational format",
        remove_columns=dataset['train'].column_names
    )


    print(dataset)

    dataset_train = dataset['train'].shuffle()
    dataset_val = dataset['validation'].shuffle()

    # Tokenize Dataset
    logger.info(f"Tokenizing the dataset")
    formatting_func = get_formatting_func_from_dataset(dataset_train, tokenizer)
    dataset_train = _prepare_non_packed_dataloader(
        args,
        tokenizer,
        dataset_train,
        "text",
        args.max_seq_length,
        formatting_func=formatting_func,
        add_special_tokens=True,
        remove_unused_columns=True,
    )

    dataset_val = _prepare_non_packed_dataloader(
        args,
        tokenizer,
        dataset_val,
        "text",
        args.max_seq_length,
        formatting_func=formatting_func,
        add_special_tokens=True,
        remove_unused_columns=True,
    )

    logger.info("Done with dataset!")
    return dataset_train, dataset_val



def main():
    args = parse_args()
    apply_seed(args.random_seed)
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    wandb.init(project="codejudge", config=vars(args), name=f"codejudge-{args.run_name}-{time.strftime('%Y%m%d-%H%M%S')}")


    pathlib.Path(args.log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        filename=args.log_dir+'-output-logs.txt', filemode="w+",
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
    )
    # Get the root logger
    logger = logging.getLogger("rich")
    # Create a handler for stdout using RichHandler (or StreamHandler if not using Rich)
    console_handler = RichHandler(rich_tracebacks=True)  # You can use `logging.StreamHandler()` if not using `RichHandler`
    console_handler.setLevel(logging.INFO)
    # Set the same format for the console handler
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    console_handler.setFormatter(formatter)
    # Add the console handler to the logger
    logger.addHandler(console_handler)


    logger.info(f"Loading model and tokenizer for {args.llm_path} with max sequence length {args.max_seq_length}.")


    # BitsAndBytesConfig int-4 config
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=args.use_nested_quant,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )
    if compute_dtype == torch.float16 and args.load_in_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.llm_path,
        attn_implementation=args.attn_implementation,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.llm_path, device_map=device_map, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Loading training and validation datasets.
    logger.info(f"Loading training {args.train_dataset_name} and validation dataset {args.validation_dataset_name}")
    train_ds, validation_ds = sft_load_dataset(args, logger, tokenizer)

    peft_config = None
    if args.is_peft:
        # Do model patching and add fast LoRA weights
        logger.info("Loading the model in PEFT mode.")
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )

    logger.info("Creating SFT trainer!")

    class CustomSFTTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Forward pass with labels to compute the loss
            outputs = model(**inputs, labels=inputs["input_ids"], return_dict=True)
            loss = outputs.loss  # Extract only the loss

            if return_outputs:
                return loss, outputs
            else:
                return loss


    fsdp_config={
            'fsdp_activation_checkpointing': True,
            'fsdp_auto_wrap_policy': 'TRANSFORMER_BASED_WRAP',
            'fsdp_backward_prefetch_policy': 'BACKWARD_PRE',
            'fsdp_cpu_ram_efficient_loading': True,
            'fsdp_forward_prefetch': False,
            'fsdp_offload_params': True,
            'fsdp_sharding_strategy': 1,
            'fsdp_state_dict_type': 'SHARDED_STATE_DICT',
            'fsdp_sync_module_states': True,
            'fsdp_use_orig_params': True,
            # Gemma
            # "fsdp_transformer_layer_cls_to_wrap": ["GemmaDecoderLayer"],
            # "xla": True,
            # "xla_fsdp_grad_ckpt": True
        }

    sft_config = SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            report_to="wandb",
            save_total_limit=args.save_total_limit,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            # load_best_model_at_end=True,
            num_train_epochs=args.num_train_epochs,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            logging_dir=args.log_dir,
            output_dir=args.output_loc,
            fsdp_config=fsdp_config,
            dataloader_drop_last = True,  # Required for SPMD.
            # fsdp="full_shard",
            gradient_checkpointing=True,
            optim=args.optim,
            seed=args.random_seed,
            ddp_find_unused_parameters=False,
            ddp_timeout=100000,
            run_name=args.run_name,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            #dataset_text_field="text",
            dataset_batch_size=args.dataset_batch_size,
            max_seq_length=args.max_seq_length,
            dataset_num_proc=args.dataset_num_proc,
            dataset_kwargs={"skip_prepare_dataset":True},
            # metric_for_best_model="eval_loss",
            # greater_is_better=False,
        )


    if args.use_custom_loss:
        trainer = CustomSFTTrainer(
                model=model,
                args=sft_config,
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
                train_dataset=train_ds,
                eval_dataset=validation_ds,
                tokenizer=tokenizer,
                peft_config=peft_config,
        )
    else:
        trainer = SFTTrainer(
                model=model,
                args=sft_config,
                # data_collator=transformers.DataCollatorForSeq2Seq(
                #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                # ),
                train_dataset=train_ds,
                eval_dataset=validation_ds,
                tokenizer=tokenizer,
                #packing=True,
                peft_config=peft_config,
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training complete ***")


    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(validation_ds)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        ##################################
        # Save model and create model card
        ##################################

    logger.info("*** Save model ***")
    trainer.save_model(args.output_loc)
    logger.info("*** Save tokenizer ***")
    tokenizer.save_pretrained(args.output_loc)
    logger.info(f"Model and tokenizer saved to {args.output_loc}")

    logger.info("*** Training complete ***")
    print('end')

main()
