import wandb
from types import SimpleNamespace
from datasets import load_dataset

import torch

from transformers import AutoTokenizer, TrainingArguments
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from trl import SFTTrainer

from llm_recipes.utils import get_latest_file, LLMSampleCB, token_accuracy, param_count

WANDB_PROJECT = "tinyllama"
ENTITY = "capecape"
DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"

ds = load_dataset(DATASET_NAME)["train"]

splitted_ds = ds.train_test_split(test_size=0.05, seed=42)

# around 0.5B params
model_config = SimpleNamespace(
    dim = 1024,
    multiple_of = 256,
    num_attention_heads = 16,
    num_hidden_layers = 16,
    max_seq_len=1024,
    torch_dtype=torch.bfloat16,
)

def create_tiny_llama(config):
    "Create a TinyLlama"
    hidden_dim = 4 * config.dim
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
    
    llama_config = LlamaConfig(
        hidden_size=config.dim,
        intermediate_size=hidden_dim, 
        num_hidden_layers=config.num_hidden_layers, 
        num_attention_heads=config.num_attention_heads, 
        max_position_embeddings=config.max_seq_len,
        torch_dtype=config.torch_dtype,
    )
    
    model = LlamaForCausalLM(llama_config)
    return model, llama_config

model, llama_config = create_tiny_llama(model_config)

param_count(model)

def load_tokenizer(model_id = 'meta-llama/Llama-2-7b-hf'):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = load_tokenizer()

# estimation
total_num_steps = int(1.2e12//1024)
eval_steps = 1000 # same as save steps
save_steps = eval_steps

output_dir = "/tmp/transformers"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=40,
    per_device_eval_batch_size=40,
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=total_num_steps // 100,
    max_steps=total_num_steps,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=save_steps,
)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=splitted_ds["train"],
    eval_dataset=splitted_ds["test"],
    dataset_text_field="text",
    packing=True,
    max_seq_length=1024,
    args=training_args,
)


### A handy callback!

from functools import partial
from transformers import GenerationConfig
from transformers.integrations import WandbCallback
from tqdm.auto import tqdm

def _generate(prompt, model, tokenizer, gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(tokenized_prompt, 
                                generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, tok_model_id = 'meta-llama/Llama-2-7b-hf'):
        super().__init__()
        self._log_model = 'checkpoint'
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.gen_config = GenerationConfig.from_pretrained(tok_model_id, max_new_tokens=max_new_tokens)
        tokenizer = trainer.tokenizer
        self.generate = partial(_generate, 
                                model=trainer.model, 
                                tokenizer=tokenizer, 
                                gen_config=self.gen_config)

    def log_generations_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt[-1000:])
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        self._wandb.log({"sample_predictions":records_table})
    
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self.log_generations_table(self.sample_dataset)        

wandb.init(project=WANDB_PROJECT, entity=ENTITY, job_type="train")

# we have to pass the callback to the training func!
wandb_callback = LLMSampleCB(trainer, splitted_ds["test"], num_samples=10, max_new_tokens=128)
trainer.add_callback(wandb_callback)

trainer.train()

wandb.finish()