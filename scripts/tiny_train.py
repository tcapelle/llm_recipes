import wandb
from types import SimpleNamespace
from datasets import load_dataset

import torch

from transformers import AutoTokenizer, TrainingArguments
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from trl import SFTTrainer

from llm_recipes.utils import get_latest_file, LLMSampleCB, token_accuracy, param_count, parse_args

WANDB_PROJECT = "tinyllama"
ENTITY = "capecape"
DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
LAST_CHECKPOINT = None
TOTAL_DATASET_SHARDS = 20

config = SimpleNamespace(
    resume_from_checkpoint=LAST_CHECKPOINT,  # uses LAST Checkpoint from W&B
    use_flash_attention_2=True,
    torch_compile=True,
    batch_size=48,
    learning_rate=5e-4,
    gradient_accumulation_steps=2,
    eval_steps=1000, # save model here
    dataloader_num_workers=6,
    dataset_shard=0,
    test_size=0.02,
    seed=42,
)

parse_args(config)

def get_datasets(ds_name, total_shards=1, shard=0, seed=42, test_size=0.02):
    "Let's shard so resuming is easier"
    ds = load_dataset(DATASET_NAME)["train"]
    splitted_ds = ds.shuffle(seed=seed).train_test_split(test_size=test_size, seed=seed)
    train_ds = splitted_ds["train"].shard(total_shards, shard)
    eval_ds = splitted_ds["test"]
    print(f"Train samples {len(train_ds)} (shard={shard+1}/{total_shards-1}) / Eval samples {len(eval_ds)}")
    return train_ds, eval_ds

train_ds, eval_ds = get_datasets(
    DATASET_NAME,
    total_shards=TOTAL_DATASET_SHARDS,
    shard=config.dataset_shard,
    seed=config.seed,
    test_size=config.test_size)

# around 0.3B params
model_config = SimpleNamespace(
    dim = 1024,
    multiple_of = 256,
    num_attention_heads = 16,
    num_hidden_layers = 16,
    max_seq_len=1024,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=config.use_flash_attention_2,
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


## MFU computations
def _model_flops(config, dt):
    "TODO: Refactor stuff that is computed once"
    N = sum(p.numel() for p in model.parameters())
    L, H, Q, T = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.dim//cfg.num_attention_heads, cfg.max_seq_len
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = config.batch_size * config.gradient_accumulation_steps * fwdbwd_per_iter
    return flops_per_iter * (1.0/dt)
#####################

param_count(model)

def load_tokenizer(model_id = 'meta-llama/Llama-2-7b-hf'):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = load_tokenizer()

# estimation
effective_batch_size = 1024*config.batch_size*config.gradient_accumulation_steps
total_num_steps = int(1.2e12)//effective_batch_size
print(f"\nTraining for {total_num_steps} with an effective batch_size of: {effective_batch_size}\n")

output_dir = "/tmp/transformers"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    bf16=True,
    dataloader_num_workers=config.dataloader_num_workers,
    torch_compile=config.torch_compile,
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=total_num_steps // 100,
    max_steps=total_num_steps,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=config.eval_steps,
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=config.eval_steps,
)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
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

wandb.init(project=WANDB_PROJECT, entity=ENTITY, job_type="train", tags=[f"shard_{config.dataset_shard}"])
wandb_callback = LLMSampleCB(trainer, eval_ds, num_samples=20, max_new_tokens=128)
trainer.add_callback(wandb_callback)

if isinstance(config.resume_from_checkpoint, str):
    artifact = wandb.use_artifact(LAST_CHECKPOINT, type='model')
    checkpoint_dir = artifact.download()
    trainer.train(resume_from_checkpoint=checkpoint_dir)
elif config.resume_from_checkpoint:
    print("Warning: RESUMING FROM CHECKPOINT ON YOUR COMPUTER (not W&B)")
    trainer.train(resume_from_checkpoint=True)
else:
    print("Warning: TRAINING FROM SCRATCH")
    trainer.train()