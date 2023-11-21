import argparse, wandb, time, torch
from types import SimpleNamespace
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from llm_recipes.utils import parse_args

config = SimpleNamespace(
    # model_id = 'meta-llama/Llama-2-7b-hf',
    model_id = "stas/tiny-random-llama-2",
    batch_size = 1, # what my GPU can handle, depends on how many layers are we training
    n_freeze=24, # how many layers to freeze on the model (llama 7b has 32)
    freeze_embed = False,
    gradient_checkpointing = False,
    use_lora=False,
    train_steps=10,
)
if __name__ == "__main__":
    parse_args(config)
    wandb.init(project="debug_lora", job_type="debug", tags=["7b"], config=config)

    peft_config = LoraConfig(
        r=16,  # the rank of the LoRA matrices
        lora_alpha=32, # the weight
        lora_dropout=0.1, # dropout to add to the LoRA layers
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    # freeze layers (disable gradients)
    for param in model.parameters(): param.requires_grad = False
    for param in model.lm_head.parameters(): param.requires_grad = True
    for param in model.model.layers[config.n_freeze:].parameters(): param.requires_grad = True
    if config.freeze_embed: model.model.embed_tokens.weight.requires_grad_(False)
    
    # gradient checkpointing
    if config.gradient_checkpointing: model.gradient_checkpointing_enable()

    # create a batch
    vocab = 3000 if "stas" in config.model_id else 32_000
    batch = {"input_ids": torch.randint(0, vocab, (config.batch_size, 512)).cuda(),
                "labels": torch.randint(0, vocab, (config.batch_size, 512)).cuda()}

    # maybe add LoRA
    if config.use_lora:
        model = get_peft_model(model, peft_config)
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])/1_000_000
    print(f"Trainable params: {trainable_params}M / {sum([p.numel() for p in model.parameters()])/1_000_000:.2f}M")

    model.train()
    for i in range(config.train_steps):
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
            loss = output.loss
            loss.backward()
        delta_t = time.perf_counter() - t0
        wandb.log({"time":delta_t, "it_s": config.batch_size/delta_t})

    memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    print(f"Memory usage: {memory_usage:.2f}GB")
    wandb.summary["memory_GB"] = memory_usage