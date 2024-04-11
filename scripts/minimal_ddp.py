from dataclasses import dataclass

import wandb
import torch
import simple_parsing
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import AutoTokenizer


@dataclass
class Config:
    model_id: str = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    dataset_id: str = "roneneldan/TinyStories"
    batch_size: int = 4
    eval_steps: int = 200
    save_steps: int = 200
    warmup_steps: int = 100
    num_train_epochs: int = 3
    sample_size: int = 1000
    block_size: int = 32

def get_dataset(dataset_id, tokenizer, sample=1000, test_pct=0.2, block_size=128):
    dataset = load_dataset(dataset_id)["train"]
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    dataset = (dataset.select(range(sample))
                      .map(tokenize_function, remove_columns="text", batched=True)
                      .map(lambda x: {"input_ids": x['input_ids'][:block_size], 
                                      "labels": x['input_ids'][:block_size]})
                      .train_test_split(test_pct))


    return dataset["train"], dataset["test"]




if __name__ == "__main__":

    config = simple_parsing.parse(Config)

    config = Config()

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id) 
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, eval_dataset = get_dataset(config.dataset_id, tokenizer, sample=config.sample_size)

    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        wandb.init(
            project="minimal_ddp",
            name=f"process_{accelerator.process_index}",
            group="my_fancy_experiment",
        )

    training_args = TrainingArguments(
        output_dir="./random_tiny_llama", #The output directory
        num_train_epochs=config.num_train_epochs, # number of training epochs
        per_device_train_batch_size=config.batch_size, # batch size for training
        per_device_eval_batch_size=config.batch_size*2,  # batch size for evaluation
        eval_steps = config.eval_steps, # Number of update steps between two evaluations.
        save_steps=config.save_steps, # after # steps model is saved
        warmup_steps=config.warmup_steps,# number of warmup steps for learning rate scheduler
        logging_strategy="steps",
        logging_steps=10,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()