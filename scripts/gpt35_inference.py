import wandb
from types import SimpleNamespace
from tqdm.auto import tqdm
import pandas as pd

from llm_recipes.openai import completion_with_backoff
from llm_recipes.utils import parse_args

WANDB_PROJECT = "alpaca_ft"
WANDB_ENTITY = "capecape"
INPUT_DATASET_TABLE_AT = dict(
    table_at = 'capecape/alpaca_ft/run-oxql1s2a-eval_predictions:v0', 
    table_name = "eval_predictions"
)

config = SimpleNamespace(
    max_tokens=256,
    system_prompt="You are a helpful assistant.",
    temperature=1,
    prompt_col="prompt",
    input_dataset_table_at=INPUT_DATASET_TABLE_AT,
    output_table="gpt35_results",
    num_samples=-1,  # for debug purposes
)

def generate_35(prompt, config):
    completion = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": config.system_prompt},
                  {"role": "user","content": prompt}
                 ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    
    )
    output = completion.choices[0].message.content
    # print(f"##PROMPT: {prompt}\n##COMPLETION: {output}")
    return output

def download_table(table_at, table_name):
    artifact = wandb.use_artifact(table_at, type='run_table')
    artifact_dir = artifact.download()
    table = artifact.get(table_name)
    df = pd.DataFrame(data=table.data, columns=table.columns)
    return df


if __name__ == "__main__":
    parse_args(config)
    # create a run to have lineage
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="eval", tags=["gpt-3.5"], config=config)
    
    input_df = download_table(**config.input_dataset_table_at)
    results = []
    for i, prompt in enumerate(tqdm(input_df[config.prompt_col].to_list()[:config.num_samples])):
        results.append((i, generate_35(prompt, config)))
    gpt35_df = pd.DataFrame(results, columns=["index", "generation"])

    # format to save
    gpt35_df = gpt35_df.assign(prompt=input_df[config.prompt_col])
    gpt35_df = gpt35_df[["index", "prompt", "generation"]]
    print(gpt35_df.head())
    
    gpt35_table = wandb.Table(dataframe=gpt35_df)
    
    # save our work's output!
    wandb.log({config.output_table:gpt35_table})