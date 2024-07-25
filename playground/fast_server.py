from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
from openai import OpenAI
import weave
from guard import PromptGuard, LlamaGuard
from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute

WEAVE_PROJECT = "prompt-eng/llama_405b_interceptor"
LLAMA_IP_ADDRESS = os.environ.get("LLAMA_IP_ADDRESS")
weave.init(WEAVE_PROJECT)

prompt_guard_model = PromptGuard()
llama_guard_model = LlamaGuard()

app = FastAPI()

# Set up rate limiter
mem_storage = storage.MemoryStorage()
rate_limiter = FixedWindowRateLimiter(mem_storage)
rate_limit =  RateLimitItemPerMinute(5)

rate_limit_response = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Stop hitting my endpoint!",
                "role": "assistant"
            },
            "logprobs": None
        }
    ],
    "created": 1677664795,
    "id": "chatcmpl-ratelimit",
    "model": "LLama-3.1-405B-Instruct-FP8",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 5,
        "prompt_tokens": 0,
        "total_tokens": 5
    }
}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter.hit(rate_limit, client_ip):
        return JSONResponse(
            status_code=429, content=rate_limit_response
        )
    response = await call_next(request)
    return response

@weave.op
def maybe_call_model(data):
    # call OPenai
    llama_client = OpenAI(
        base_url=f"http://{LLAMA_IP_ADDRESS}/v1",  # the endpoint IP running on vLLM
        api_key="dummy_api_key",  # the endpoint API key
    )

    @weave.op
    def call_llama(messages):
        print("Calling Llama 405B")
        completion = llama_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        messages=messages,
        )
        return completion
    
    messages = data.get("messages", [])

    prompt_guard_out = prompt_guard_model.predict(messages)
    llama_guard_out = llama_guard_model.predict(messages)
    print(f"Guard output: {prompt_guard_out}")
    print(f"Llama Guard output: {llama_guard_out}")

    # if "unsafe" in llama_guard_out:
    response = call_llama(messages).dict()
    return JSONResponse(content=response)

    # else:
    #     mock_response = {
    #         "choices": [
    #             {
    #                 "finish_reason": "stop",
    #                 "index": 0,
    #                 "message": {
    #                     "content": "You are not playing the game correctly, please try breaking LLama3.1.",
    #                     "role": "assistant"
    #                 },
    #                 "logprobs": None
    #             }
    #         ],
    #         "created": 1620000000,
    #         "id": "chatcmpl-mockid123456",
    #         "model": "LLama-3.1-405B-Instruct-FP8",
    #         "object": "chat.completion",
    #         "usage": {
    #             "completion_tokens": 13,
    #             "prompt_tokens": 0,
    #             "total_tokens": 13
    #         }
    #     }
    #     return JSONResponse(content=mock_response)


@app.post("/v1/{path:path}")
async def intercept_openai(request: Request, path: str):
    # Get the request body
    body = await request.body()
    
    # Parse the JSON body
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = {}
    
    # Log the intercepted request
    print(f"Intercepted OpenAI API call to /{path}")
    print(f"Request body: {data}")
    
    # Here you can add your custom logic, e.g., modifying the request,
    # forwarding it to OpenAI, or returning a mock response
    
    model_out = maybe_call_model(data)
    return model_out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)