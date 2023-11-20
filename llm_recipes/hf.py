from peft import LoraConfig, get_peft_model

DEFAULT_LORA_CONFIG = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16, # the weight
    lora_dropout=0.1, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
)

def create_peft_model(
        model, 
        gradient_checkpointing=False, 
        peft_config=DEFAULT_LORA_CONFIG):
    # create the LoRA config

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config