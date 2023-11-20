from types import SimpleNamespace

import wandb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset


from llm_recipes.data import create_alpaca_prompt
from llm_recipes.utils import freeze, parse_args

ALPACA_TOTAL_PACKED_SAMPLES = 11_210
WANDB_PROJECT = "alpaca_ft"
WANDB_ENTITY = "capecape"
WANDB_TAGS = ["7b", "hf_sft"]

config = SimpleNamespace(
    dataset_at='capecape/alpaca_ft/alpaca_gpt4_splitted:latest',
    model_id = 'meta-llama/Llama-2-7b-hf',
    n_freeze = 24,
    batch_size = 8,
    num_train_epochs = 3,
    freeze_embed = True,
)

# some sane defaults computations
config.gradient_accumulation_steps = 32 // config.batch_size
config.total_num_steps = config.num_train_epochs * ALPACA_TOTAL_PACKED_SAMPLES // (config.batch_size * config.gradient_accumulation_steps)
config.eval_steps = config.total_num_steps // config.num_train_epochs


def get_alpaca_ds(dataset_at):
    artifact = wandb.use_artifact(dataset_at, type='dataset')
    artifact_dir = artifact.download()
    alpaca_ds = load_dataset("json", data_dir=artifact_dir)
    train_dataset = alpaca_ds["train"]
    eval_dataset = alpaca_ds["test"]
    return train_dataset, eval_dataset

def get_train_args(config, output_dir = "./output/"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size//4,
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # num_train_epochs=config.num_train_epochs,
        max_steps=config.total_num_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        # logging strategies
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
    )
    return training_args

def main(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    freeze(model, config.n_freeze, config.freeze_embed)
    wandb.init(project=WANDB_PROJECT, 
               entity=WANDB_ENTITY, 
               job_type="train",
               tags=WANDB_TAGS,
               config=config)
    train_dataset, eval_dataset = get_alpaca_ds(config.dataset_at)
    
    # override whatever train args we may need
    training_args = get_train_args(config)
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True,
        max_seq_length=1024,
        args=training_args,
        formatting_func=create_alpaca_prompt,
    )
    trainer.train()

if __name__ == "__main__":
    parse_args(config)
    # print(config)
    main(config)