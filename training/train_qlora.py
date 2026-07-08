import os
import argparse
from pathlib import Path
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/configs/qlora.yaml")
    parser.add_argument("--smoke_test", action="store_true", help="Run only a few steps to verify the pipeline")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()

def format_instruction(example):
    prompt = (
        f"Classify the following contract clause into one of the specified categories.\n\n"
        f"Categories: Governing Law, Anti-Assignment, Cap On Liability, License Grant, "
        f"Audit Rights, Termination For Convenience, Exclusivity, Renewal Term, Insurance, "
        f"Ip Ownership Assignment, Change Of Control, Non-Compete, Uncapped Liability, "
        f"Revenue/Profit Sharing, None.\n\n"
        f"Clause: {example['input']}\n\n"
        f"Category: "
    )
    return {"text": f"{prompt}{example['output']}"}

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    train_dataset = load_dataset("json", data_files=config["train_file"], split="train")
    val_dataset = load_dataset("json", data_files=config["val_file"], split="train")
    
    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(40, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(20, len(val_dataset))))
    
    train_dataset = train_dataset.map(format_instruction)
    val_dataset = val_dataset.map(format_instruction)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    if args.smoke_test:
        training_args_kwargs = {
            "max_steps": 10,
            "save_strategy": "steps",
            "save_steps": 5,
            "evaluation_strategy": "steps",
            "eval_steps": 5,
        }
    else:
        training_args_kwargs = {
            "num_train_epochs": config["epochs"],
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
        }

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        logging_steps=5 if args.smoke_test else 10,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="wandb" if config.get("use_wandb", False) else "none",
        run_name=config.get("run_name", "contractlens-qlora"),
        **training_args_kwargs
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
        peft_config=peft_config,
    )
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint else None)
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()
