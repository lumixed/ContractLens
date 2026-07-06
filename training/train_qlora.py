import os
import argparse
from pathlib import Path
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/configs/qlora.yaml")
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
    
    train_dataset = train_dataset.map(format_instruction)
    val_dataset = val_dataset.map(format_instruction)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=config["epochs"],
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none"
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
    
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()
