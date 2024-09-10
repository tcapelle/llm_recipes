import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import json
import httpx
import logging
import os
import weave
from dataclasses import dataclass
import time
from collections import deque

from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute

import simple_parsing

app = FastAPI()

@dataclass
class Config:
    rate_limit_requests_per_minute: int = 20
    rate_limit_error_message: str = "Rate limit exceeded. Please try again later."
    mistral_api_endpoint: str = "https://api.mistral.ai/v1/chat/completions"
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY")
    timeout: int = 30
    max_concurrent_requests: int = 20  # New parameter for semaphore
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "INFO"
    weave_project: str = "prompt-eng/interceptor-mistral-hackercup"
    stats_window_size: int = 3600  # 1 hour window for statistics

config = simple_parsing.parse(Config)

# Set up logging
logger = logging.getLogger('uvicorn.error')
logging.basicConfig(level=config.log_level)
weave.init(config.weave_project)

# Set up rate limiter
mem_storage = storage.MemoryStorage()
rate_limiter = FixedWindowRateLimiter(mem_storage)
rate_limit = RateLimitItemPerMinute(config.rate_limit_requests_per_minute)

rate_limit_response = {
    "error": {
        "message": config.rate_limit_error_message,
        "type": "rate_limit_error",
        "param": None,
        "code": "rate_limit_exceeded"
    }
}

# Create a semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(config.max_concurrent_requests)

class Stats:
    def __init__(self, window_size):
        self.window_size = window_size
        self.request_times = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)

    def record_request(self, timestamp, tokens):
        self.request_times.append(timestamp)
        self.token_counts.append((timestamp, tokens))

    def calculate_request_stats(self):
        if not self.request_times:
            return 0, 0
        
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        relevant_requests = [t for t in self.request_times if t > one_minute_ago]
        requests_last_minute = len(relevant_requests)
        
        rps = requests_last_minute / 60
        rpm = requests_last_minute
        
        return rps, rpm

    def calculate_token_stats(self):
        if not self.token_counts:
            return 0, 0
        
        current_time = time.time()
        one_minute_ago = current_time - 60
        one_hour_ago = current_time - 3600
        
        tokens_last_minute = sum(count for t, count in self.token_counts if t > one_minute_ago)
        tokens_last_hour = sum(count for t, count in self.token_counts if t > one_hour_ago)
        
        return tokens_last_minute, tokens_last_hour

# Initialize Stats
stats = Stats(config.stats_window_size)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter.hit(rate_limit, client_ip):
        return JSONResponse(status_code=429, content=rate_limit_response)
    response = await call_next(request)
    return response

@weave.op
async def log_data(ip, input, output):
    return 

@app.post("/v1/{path:path}")
async def forward_request(request: Request, path: str):
    # Get the request body
    body = await request.body()

    headers = request.headers
    # headers["authorization"] = f"Bearer {config.mistral_api_key}"
    
    # Parse the JSON body
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    
    # Forward the request to the Mistral API
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {config.mistral_api_key}"
        }
        # Log the full request details before sending
        print("-" * 100)
        print(f"Headers: \n{json.dumps(headers, indent=4)}")
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        print(f"Data being sent: \n{json_data}")
        async with semaphore:  # Use semaphore to limit concurrent requests
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    data=json_data,
                    headers=headers,
                    timeout=config.timeout,
                )
                print("-" * 100)
                print(f"Response: \n{json.dumps(response.json(), indent=4)}")
                print("=" * 100)
            logger.info(f"Forwarding request to Mistral API")
            output =  JSONResponse(content=response.json(), status_code=response.status_code)
            await log_data(request.client.host, data, response.json())
            # Record request time and token count
            current_time = time.time()
            total_tokens = response.json().get('usage', {}).get('total_tokens', 0)
            stats.record_request(current_time, total_tokens)
            
            # Calculate and log statistics
            rps, rpm = stats.calculate_request_stats()
            tokens_per_minute, tokens_per_hour = stats.calculate_token_stats()
            logger.info(f"STATS:\nCurrent RPS: {rps:.2f}\nRPM: {rpm}\nTokens/min: {tokens_per_minute}\nTokens/hour: {tokens_per_hour}")
            
            return output
    except Exception as e:
        logger.error(f"Error calling Mistral API: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    uvicorn.run(app, host=config.server_host, port=config.server_port)