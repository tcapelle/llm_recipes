## Different ways of creating an Instruction dataset
## https://wandb.ai/capecape/alpaca_ft/reports/Exploring-Collation-methods-Different-ways-to-construct-a-batch--Vmlldzo1ODczNjE5

import wandb
from .utils import load_jsonl

DEFAULT_ALPACA_SPLIT = 'capecape/alpaca_ft/alpaca_gpt4_splitted:v4'

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

def create_alpaca_prompt_with_response(row):
    instruct = _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)
    return instruct + row["output"]

def get_alpaca_split(dataset_at = DEFAULT_ALPACA_SPLIT):
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

def collate_and_pad(tokenizer, mask_prompt=False):
    "Collate that pads to longest sample, very inneficient"

    if mask_prompt:
        def _inner(examples):
            input_ids = tokenizer([x["prompt"]+x["output"]+tokenizer.eos_token for x in examples], 
                                  return_tensors='pt', padding="longest")['input_ids']
            prompts = tokenizer([x["prompt"] for x in examples])
            for prompt, input_id in zip(prompts, input_ids):
                input_id[:len(prompt)] = -100  # mask instructions
            batch = {'input_ids': input_ids[:, :-1], 'labels': input_ids[:, 1:]}
            return batch
    else:        
        def _inner(examples):
            examples = [x["prompt"]+x["output"]+tokenizer.eos_token for x in examples]
            batch_size = len(examples)
            input_ids = tokenizer(examples, return_tensors='pt', padding="longest")['input_ids']
            batch = {'input_ids': input_ids[:, :-1], 'labels': input_ids[:, 1:]}
            return batch    
    
    return _inner

def pad_to_len(seq, max_seq_len, pad_token_id):
    "Pad a `seq` to `max_seq_len` with `pad_token_id`"
    if len(seq) < max_seq_len:
        seq = seq + [pad_token_id] * (max_seq_len - len(seq))
    return seq

def standard_packing(dataset, tokenizer, max_seq_len):
    """Pack multiple examples into a sequences of lenght `max_seq_len`
    May split instructions at the end/begining of sequences."""
    tkds_ids = tokenizer([s["prompt"] + s["output"] for s in dataset])["input_ids"]
    
    all_token_ids = []
    for tokenized_input in tkds_ids:
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])
    
    print(f"Total number of tokens: {len(all_token_ids)}")
    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len):
        input_ids = all_token_ids[i : i + max_seq_len]
        if len(input_ids) == max_seq_len:  # drop last
            packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})
    return packed_ds

def pad_packing(dataset, tokenizer, max_seq_len):
    """Pack multiple examples into a sequences of lenght `max_seq_len`.
    Does not split instructions, pads with EOS token up to `max_seq_len`.
    """
    pad_token=tokenizer.pad_token_id
    tkds_ids = tokenizer([s["prompt"] + s["output"] for s in dataset])["input_ids"]    

    packed_ds = [] 
    current_seq = []
    for tokenized_input in tkds_ids:
        if len(current_seq) < max_seq_len - len(tokenized_input):
            current_seq.extend(tokenized_input + [tokenizer.eos_token_id])
        else:
            input_ids = pad_to_len(current_seq, max_seq_len, pad_token)
            packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})

            # we start next sequence
            current_seq = tokenized_input + [tokenizer.eos_token_id]
    return packed_ds

def masking_and_packing(dataset, tokenizer, max_seq_len):
    """Pack multiple examples into a sequences of lenght `max_seq_len`.
    Masks prompt tokens with -100 so cross entropy ignores them"""
    pad_token=tokenizer.pad_token_id
    prompt_ids = tokenizer([s["prompt"] for s in dataset])["input_ids"]
    outputs_ids = tokenizer([s["output"] for s in dataset], add_special_tokens=False)["input_ids"]

    all_token_ids = []
    all_labels_ids = []
    for prompt, output in zip(prompt_ids, outputs_ids):
        all_token_ids.extend(prompt + output + [tokenizer.eos_token_id])
        all_labels_ids.extend([-100]*len(prompt) + output + [tokenizer.eos_token_id])

    assert len(all_token_ids) == len(all_labels_ids), "Error on tokenizing"
    
    print(f"Total number of tokens: {len(all_token_ids)}")
    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len):
        input_ids = all_token_ids[i : i + max_seq_len]
        label_ids = all_labels_ids[i : i + max_seq_len]
        if len(input_ids) == max_seq_len:  # drop last
            packed_ds.append({"input_ids": input_ids[:-1], "labels": label_ids[1:]})
    return packed_ds

def pad_mask_packing(dataset, tokenizer, max_seq_len):
    """Pack multiple examples into a sequences of lenght `max_seq_len`.
    Masks prompt tokens with -100 so cross entropy ignores them.
    Does not split instructions, pads with EOS token up to `max_seq_len`"""
    pad_token=tokenizer.pad_token_id
    prompt_ids = tokenizer([s["prompt"] for s in dataset])["input_ids"]
    outputs_ids = tokenizer([s["output"] for s in dataset], add_special_tokens=False)["input_ids"]

    packed_ds = [] 
    current_seq_inputs = []
    current_seq_outputs = []
    for prompt, output in zip(prompt_ids, outputs_ids):
        example = prompt + output + [tokenizer.eos_token_id]
        label = [-100]*len(prompt) + output + [tokenizer.eos_token_id]
        if len(current_seq_inputs) < max_seq_len - len(example):
            current_seq_inputs.extend(example)
            current_seq_outputs.extend(label)
        else:
            input_ids = pad_to_len(current_seq_inputs, max_seq_len, pad_token)
            label_ids = pad_to_len(current_seq_outputs, max_seq_len, pad_token)
            packed_ds.append({"input_ids": input_ids[:-1], "labels": label_ids[1:]})
            
            # create next example
            current_seq_inputs = example
            current_seq_outputs = label
    return packed_ds


def create_packed_datasets(dataset, tokenizer, max_seq_length=1024, formatting_func=lambda text: text):
    "Same as standard_pack but using trl"
    try:
        from trl.trainer import ConstantLengthDataset
    except:
        print("Install `trl` with `pip install trl`")
    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, formatting_func)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=formatting_func,
        infinite=False,
        seq_length=max_seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=formatting_func,
        infinite=False,
        seq_length=max_seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset

def chars_token_ratio(dataset, tokenizer, formatting_func, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = formatting_func(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens