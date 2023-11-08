from functools import partial
from time import perf_counter
from types import SimpleNamespace

import pandas as pd
import wandb
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llm_recipes.utils import parse_args, load_model_from_artifact
from llm_recipes.data import get_alpaca_split, create_alpaca_prompt

WANDB_PROJECT = "alpaca_ft"
ENTITY = "reviewco"
DATASET_AT = 'capecape/alpaca_ft/alpaca_gpt4_splitted:v4'
MODEL_AT = 'reviewco/alpaca_ft/1adjrp3l_meta-llama_Llama-2-7b-hf:v0'

TORCH_DTYPES = {"bf16":torch.bfloat16, "fp16":torch.float16, "fp32":torch.float}

config = SimpleNamespace(
    wandb_project=WANDB_PROJECT,
    entity=ENTITY,
    dataset_at=DATASET_AT,
    model_at=MODEL_AT,
    use_cache=True,
    device_map="auto",
    torch_dtype="bf16",
    ## Generation params
    num_eval_samples=250, # we only run on the first samples of the eval_dataset
    temperature=0.6,
    max_new_tokens=256, # how many tokens to generate as a response
)

def _generate(prompt, model, tokenizer, gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    with torch.inference_mode():
        t0 = perf_counter()
        output = model.generate(tokenized_prompt, 
                                generation_config=gen_config)
        generation_ids = output[0][len(tokenized_prompt[0]):]
        num_gen_tokens = len(generation_ids)
        total_time = perf_counter() - t0
        generation = tokenizer.decode(generation_ids, skip_special_tokens=True)
        return dict(generation=generation, generation_ids=generation_ids.tolist(), total_time=total_time, num_gen_tokens=num_gen_tokens)
    
def evaluate(examples, model, tokenizer, gen_config):
    sample_out = _generate(examples[0]["prompt"], model, tokenizer, gen_config)
    columns = ["prompt", "label"] + list(sample_out.keys()) + ["temperature", "max_new_tokens"]
    data = []
    for example in tqdm(examples, leave=False):
        prompt, label = example["prompt"], example["output"]
        output = _generate(prompt, model, tokenizer, gen_config)
        data.append((prompt, label, *list(output.values()), config.temperature, config.max_new_tokens))
    return data, columns
    

if __name__=="__main__":
    parse_args(config)
    wandb.init(project=config.wandb_project, 
               entity=config.entity, 
               job_type="inference", 
               config=config)
    config = wandb.config
    
    _, eval_dataset = get_alpaca_split(dataset_at=config.dataset_at)
    
    model, tokenizer, artifact_dir = load_model_from_artifact(
        config.model_at, 
        use_cache=config.use_cache,
        device_map=config.device_map,
        torch_dtype=TORCH_DTYPES[config.torch_dtype],
    )
    model.eval();
    
    # inference stuff
    gen_config = GenerationConfig.from_pretrained(artifact_dir, 
                                                  temperature=config.temperature,
                                                  max_new_tokens=config.max_new_tokens)
    config.gen_config = gen_config  # we save this 
    wandb.config.update(config)
    
    print("Running inference on your model!")
    eval_samples = eval_dataset[:config.num_eval_samples]
    data, columns = evaluate(eval_samples, model=model, tokenizer=tokenizer, gen_config=gen_config)
   
    df = pd.DataFrame(data=data, columns=columns)
    print(df.head())
    df.to_csv(f"{wandb.run.id}_results.csv")
    table = wandb.Table(dataframe=df)
    wandb.log({"eval_predictions": table})






















