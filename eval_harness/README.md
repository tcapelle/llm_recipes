Test Eval Harness:

```
python -m lm_eval --model hf \
    --model_args pretrained=mistralai/Mistral-7B-v0.1,,dtype="bfloat16" \
    --tasks hellaswag,arc_challenge,truthfulqa,mmlu,winogrande \
    --device cuda:0 \
    --batch_size 1
    --limit 10

python run.py --model_path "llm_surgery/shearllama/model-5q2vwifx:v0" --tasks "hellaswag,arc_challenge,truthfulqa,mmlu,winogrande" --limit 10 --log_samples

```

Using the Launcher in W&B
```
{
    "args": [
        "--model_path",
        "llm_surgery/shearllama/model-5q2vwifx:v0",
        "--tasks",
        "hellaswag,arc_challenge,mmlu,winogrande",
        "--limit",
        "10",
        "--log_samples"
    ],
    "run_config": {},
    "entry_point": []
}

```


docker run --rm -e WANDB_BASE_URL=https://api.wandb.ai -e WANDB_API_KEY -e WANDB_PROJECT=shearllama -e WANDB_ENTITY=llm_surgery -e WANDB_LAUNCH=True -e WANDB_RUN_ID=o6kr4lau -e WANDB_DOCKER=eval_harness_at_pre -e WANDB_USERNAME=capecape -e WANDB_LAUNCH_QUEUE_NAME='Eval Harness A10' -e WANDB_LAUNCH_QUEUE_ENTITY=llm_surgery -e WANDB_LAUNCH_TRACE_ID=UnVuUXVldWVJdGVtOjUzOTgyMjMyNw== -e WANDB_CONFIG='{}' -e WANDB_ARTIFACTS='{"_wandb_job": "llm_surgery/shearllama/eval_harness_at:latest"}' --rm --gpus all --ulimit stack=67108864 --ulimit memlock=-1 --volume /home/ubuntu/.cache/huggingface:/root/.cache/huggingface --shm-size 10g --privileged eval_harness_at_pre --model_path llm_surgery/shearllama/model-5q2vwifx:v0 --tasks hellaswag,arc_challenge,mmlu,winogrande --limit 10 log_samples