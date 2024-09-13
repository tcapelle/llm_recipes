import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import json
import httpx
import logging
import os
import openai
import weave
from dataclasses import dataclass
import time
from collections import deque, Counter

from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute

import simple_parsing


client = openai.AsyncOpenAI()

app = FastAPI()

@dataclass
class Config:
    rate_limit_requests_per_minute: int = 1
    max_concurrent_requests: int = 2  # New parameter for semaphore
    rate_limit_error_message: str = "Rate limit exceeded. Please try again later."
    server_host: str = "0.0.0.0"
    server_port: int = 8010
    log_level: str = "INFO"
    weave_project: str = "prompt-eng/interceptor-openai"
    stats_window_size: int = 3600  # 1 hour window for statistics
    verbose: bool = False
    supported_models = ["o1-preview", "o1-mini"]

config = simple_parsing.parse(Config)

# Set up logging
logger = logging.getLogger('interceptor')
logging.basicConfig(level=config.log_level)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        self.user_requests = deque(maxlen=window_size)
        self.unique_users = Counter()

    def record_request(self, timestamp, tokens, user_ip):
        self.request_times.append(timestamp)
        self.token_counts.append((timestamp, tokens))
        self.user_requests.append((timestamp, user_ip))
        self.unique_users[user_ip] += 1

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

    def calculate_unique_users(self, time_window=3600):
        current_time = time.time()
        cutoff_time = current_time - time_window
        recent_users = set(ip for t, ip in self.user_requests if t > cutoff_time)
        return len(recent_users)

    async def print_stats_periodically(self):
        while True:
            await asyncio.sleep(20)  # Wait for 20 seconds
            rps, rpm = self.calculate_request_stats()
            tokens_per_minute, tokens_per_hour = self.calculate_token_stats()
            unique_users_hour = self.calculate_unique_users(3600)  # Last hour
            unique_users_day = self.calculate_unique_users(86400)  # Last 24 hours
            
            print("=" * 100)
            logger.info(f"PERIODIC STATS:")
            logger.info(f"Current RPS: {rps:.2f}")
            logger.info(f"RPM: {rpm}")
            logger.info(f"Tokens/min: {tokens_per_minute}")
            logger.info(f"Tokens/hour: {tokens_per_hour}")
            logger.info(f"Unique users (last hour): {unique_users_hour}")
            logger.info(f"Unique users (last 24 hours): {unique_users_day}")
            print("=" * 100)

# Initialize Stats
stats = Stats(config.stats_window_size)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter.hit(rate_limit, client_ip):
        return JSONResponse(status_code=429, content=rate_limit_response)
    response = await call_next(request)
    return response


@app.post("/v1/{path:path}")
async def forward_request(request: Request, path: str):
    # Get the request body
    body = await request.body()

    # Parse the JSON body
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    
    try:
        model = data['model']
        if model not in config.supported_models:
            return JSONResponse(status_code=400, content={"error": f"Model {model} not supported, supported models: {config.supported_models}"})
        logger.info(f"Forwarding request to model {model}")
        response = await client.chat.completions.create(**data)
        user_ip = request.client.host
        logger.info(f"Serving request from {user_ip} with {response.usage.total_tokens} tokens")
        return JSONResponse(content=response.dict())
    except Exception as e:
        logger.error(f"Error calling the API: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    uvicorn.run(app, host=config.server_host, port=config.server_port)