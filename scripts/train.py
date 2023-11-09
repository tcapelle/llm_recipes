import wandb
wandb.require("nexus")

import json, random
from types import SimpleNamespace
from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from llm_recipes.utils import (
    load_jsonl, 
    save_model, 
    Accuracy, 
    to_gpu, 
    flat_cross_entropy, 
    parse_args, 
    freeze
)
from llm_recipes.data import (
    standard_packing, 
    pad_packing, 
    masking_and_packing, 
    pad_mask_packing,
    collate_and_pad
)

WANDB_PROJECT = "alpaca_ft"
ENTITY = "reviewco"
DATASET_AT = 'capecape/alpaca_ft/alpaca_gpt4_splitted:v4'
TAGS = ["7b"]


config = SimpleNamespace(
    model_id='meta-llama/Llama-2-7b-hf',
    dataset_name="alpaca-gpt4",
    seed=42,
    precision="bf16",  # faster and better than fp16, requires new GPUs
    n_freeze=24,  # How many layers we don't train, LLama 7B has 32.
    lr=1e-4,
    n_eval_samples=10, # How many samples to generate on validation
    max_seq_len=1024, # Lenght of the sequences to pack
    epochs=3,  # we do 3 pasess over the dataset.
    gradient_accumulation_steps=2,  # evey how many iterations we update the gradients, simulates larger batch sizes
    batch_size=16,  # what my GPU can handle, depends on how many layers are we training  
    log_model=False,  # upload the model to W&B?
    gradient_checkpointing = True,  # saves even more memory
    freeze_embed = True,  # why train this? let's keep them frozen ❄️
    max_new_tokens=256, # for generations
    pad=False,  # pad sequences to `max_sequence_len`
    mask_prompt=False, # mask input prompt
    packing=True, # pack for efficient training
    wandb_project=WANDB_PROJECT,
    entity=ENTITY,
    tags=TAGS,
)
parse_args(config)

def _prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)

def _prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

def create_alpaca_prompt(row):
    return _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)


def get_dataset(dataset_at = DATASET_AT):
    artifact = wandb.use_artifact(dataset_at, type='dataset')
    dataset_dir = artifact.download()
    
    train_dataset = load_jsonl(f"{dataset_dir}/alpaca_gpt4_train.jsonl")
    eval_dataset = load_jsonl(f"{dataset_dir}/alpaca_gpt4_eval.jsonl")

    def _format_dataset(dataset):
        "No EOS token yet"
        return [{"prompt":create_alpaca_prompt(row), 
                 "output": row["output"]} for row in dataset]
        
    train_dataset = _format_dataset(train_dataset)
    eval_dataset = _format_dataset(eval_dataset)
    print(train_dataset[0])
    return train_dataset, eval_dataset

def packing_func(pad=False, mask_prompt=False):
    "Packing method to use!"
    if pad and mask_prompt:
        return pad_mask_packing
    elif pad and not mask_prompt:
        return pad_packing
    elif not pad and mask_prompt:
        return masking_and_packing
    else:
        return standard_packing

run = wandb.init(project=config.wandb_project, 
                 entity=config.entity, 
                 job_type="train", 
                 tags=config.tags, 
                 config=config)
train_dataset, eval_dataset = get_dataset()

tokenizer = AutoTokenizer.from_pretrained(config.model_id)
tokenizer.pad_token = tokenizer.eos_token

if config.packing:
    pack = packing_func(config.pad, config.mask_prompt)
    proc_train_dataset = pack(train_dataset, tokenizer, config.max_seq_len)
    proc_eval_dataset = pack(eval_dataset, tokenizer, config.max_seq_len)
    collate_fn = default_data_collator
else:
    collate_fn = collate_and_pad(tokenizer, mask_prompt=config.mask_prompt)
    proc_train_dataset = train_dataset
    proc_eval_dataset = eval_dataset
    print("No packing, we are going to train for long!")

print(f"Final number of train samples: {len(train_dataset)}")

torch.manual_seed(config.seed)# I have an A100 GPU with 40GB of RAM 😎

train_dataloader = DataLoader(
    proc_train_dataset,
    batch_size=config.batch_size,
    collate_fn=collate_fn, 
)

eval_dataloader = DataLoader(
    proc_eval_dataset,
    batch_size=config.batch_size,
    collate_fn=collate_fn,
    shuffle=False,
)
# Update with the size of the dataloader now that we know it
config.total_train_steps = config.epochs * len(train_dataloader) // config.gradient_accumulation_steps
config.eval_every = config.total_train_steps//10. # we evaluate every 1/10th of the total train steps.

def prepare_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map=0,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    param_count(model)
    freeze(model, config.n_freeze, config.freeze_embed)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    print("\nFinal param count:")
    param_count(model)
    return model
        
def param_count(m):
    params = sum([p.numel() for p in m.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in m.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")

model = prepare_model(config)
gen_config = GenerationConfig.from_pretrained(config.model_id, max_new_tokens=config.max_new_tokens)
config.gen_config = gen_config  # we save this 
wandb.config.update(config)

optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9,0.99), eps=1e-5)
scheduler = get_cosine_schedule_with_warmup(
    optim,
    num_training_steps=config.total_train_steps,
    num_warmup_steps=config.total_train_steps // 10,
)

def generate(prompt, tokenizer):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(tokenized_prompt, 
                                generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

def prompt_table(examples, log=False, table_name="predictions"):
    table = wandb.Table(columns=["prompt", "generation", "concat", "output", "max_new_tokens", "temperature", "top_p"])
    for example in tqdm(examples, leave=False):
        prompt, label = example["prompt"], example["output"]
        out = generate(prompt, tokenizer)
        table.add_data(prompt, out, prompt+out, label, gen_config.max_new_tokens, 
                       gen_config.temperature, gen_config.top_p)
    if log:
        wandb.log({table_name:table})
    return table


@torch.no_grad()
def validate(samples=eval_dataset):
    model.eval();
    eval_acc = Accuracy()
    for step, batch in enumerate(pbar:=tqdm(eval_dataloader, leave=False)):
        pbar.set_description(f"doing validation")
        batch = to_gpu(batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = flat_cross_entropy(out.logits, batch["labels"])
        eval_acc.update(out.logits, batch["labels"])
    # we log results at the end
    wandb.log({"eval_loss": loss.item(),
               "eval_accuracy": eval_acc.compute()})
    prompt_table(samples[:config.n_eval_samples], log=True)
    model.train();


acc = Accuracy()
model.train()
train_step = 0
for epoch in tqdm(range(config.epochs)):
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = to_gpu(batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = flat_cross_entropy(out.logits, batch["labels"]) / config.gradient_accumulation_steps
        if step%config.gradient_accumulation_steps == 0:
            # we can log the metrics to W&B
            wandb.log({"train_loss": loss.item() * config.gradient_accumulation_steps,
                       "train_accuracy": acc.update(out.logits, batch["labels"]),
                       "lr": scheduler.get_last_lr()[0]})
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            train_step += 1

            if train_step%config.eval_every==0 or train_step%(config.total_train_steps-1)==0:
                validate(eval_dataset)    
# we save the model checkpoint at the end
save_model(model, model_name=config.model_id.replace("/", "_"), models_folder="models/", log=config.log_model)
wandb.finish()