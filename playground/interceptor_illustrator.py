import asyncio
import re
import io
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import httpx
import logging
import openai
from pathlib import Path
import sys
from dataclasses import dataclass
import time
from collections import deque, Counter
from contextlib import asynccontextmanager
from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute

import simple_parsing

from illustrator import IllustratePayload

client = openai.AsyncOpenAI()

@dataclass
class Config:
    openai_model: str = "gpt-4o"
    fal_model: str = "fal-ai/flux-pro"
    rate_limit_requests_per_minute: int = 10
    max_concurrent_requests: int = 5  # New parameter for semaphore
    rate_limit_error_message: str = "Rate limit exceeded. Please try again later."
    server_host: str = "0.0.0.0"
    server_port: int = 8020
    log_level: str = "INFO"
    weave_project: str = "interceptor-illustrator"
    stats_window_size: int = 3600  # 1 hour window for statistics
    verbose: bool = False
    python_executable: str = "illustrator.py"
    dummy: bool = False

config = simple_parsing.parse(Config)

# Set up logging
logger = logging.getLogger('interceptor')
logging.basicConfig(level=config.log_level)
logging.getLogger("httpx").setLevel(logging.WARNING)

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
        self.image_counts = deque(maxlen=window_size)
        self.user_requests = deque(maxlen=window_size)
        self.unique_users = Counter()

    def record_request(self, timestamp, images, user_ip):
        self.request_times.append(timestamp)
        self.image_counts.append((timestamp, images))
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

    def calculate_image_stats(self):
        if not self.image_counts:
            return 0, 0
        
        current_time = time.time()
        one_minute_ago = current_time - 60
        one_hour_ago = current_time - 3600
        
        images_last_minute = sum(count for t, count in self.image_counts if t > one_minute_ago)
        images_last_hour = sum(count for t, count in self.image_counts if t > one_hour_ago)
        
        return images_last_minute, images_last_hour

    def calculate_unique_users(self, time_window=3600):
        current_time = time.time()
        cutoff_time = current_time - time_window
        recent_users = set(ip for t, ip in self.user_requests if t > cutoff_time)
        return len(recent_users)

    async def print_stats_periodically(self):
        while True:
            await asyncio.sleep(20)  # Wait for 20 seconds
            rps, rpm = self.calculate_request_stats()
            images_per_minute, images_per_hour = self.calculate_image_stats()
            unique_users_hour = self.calculate_unique_users(3600)  # Last hour
            unique_users_day = self.calculate_unique_users(86400)  # Last 24 hours
            
            print("=" * 100)
            logger.info(f"PERIODIC STATS:")
            logger.info(f"Current RPS: {rps:.2f}")
            logger.info(f"RPM: {rpm}")
            logger.info(f"Images/min: {images_per_minute}")
            logger.info(f"Images/hour: {images_per_hour}")
            logger.info(f"Unique users (last hour): {unique_users_hour}")
            logger.info(f"Unique users (last 24 hours): {unique_users_day}")
            print("=" * 100)

# Initialize Stats
stats = Stats(config.stats_window_size)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background task to print stats periodically
    stats_task = asyncio.create_task(stats.print_stats_periodically())
    yield
    # Shutdown: Cancel the background task
    stats_task.cancel()
    try:
        await stats_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter.hit(rate_limit, client_ip):
        return JSONResponse(status_code=429, content=rate_limit_response)
    response = await call_next(request)
    return response

async def run_python(program: Path, *args, timeout: float = 200):
    """
    Run a Python program with the given input file and output file.
    """
    if config.verbose:
        print(f"Executing {sys.executable} {program} with args: {args}\n")
        print("=" * 100)

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            program,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Program execution timed out after {timeout} seconds")

        if process.returncode != 0:
            raise RuntimeError(f"Program execution failed: {stderr.decode()}")

        return stdout.decode(), stderr.decode()

    except Exception as e:
        raise RuntimeError(f"Error running Python program: {str(e)}")


async def generate_illustration(payload: IllustratePayload):
    # Run the python script
    args = [f"--{k}={v}" for k,v in payload.model_dump().items()]
    args += ["--dummy"] if config.dummy else []
    if config.verbose:
        print(f"Running {config.python_executable} with args:")
        for arg in args:
            print(f"  {arg}")
        print("=" * 100)
    return await run_python(
        Path(config.python_executable),
        *args,
    )

def get_weave_link(stdout: str):
    "Use regex to parse everything between two newlines and \\n🍩 and \\n"
    match = re.search(r'\n🍩(.*?)\n', stdout, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def get_images_dict(stdout: str):
    image_base64_dict = json.loads(stdout.split("<img>")[-1].split("</img>")[-2])
    return image_base64_dict

@app.post("/illustrate/{path:path}")
async def forward_request(request: Request, path: str):
    """
    Generate an illustration for a story.
    """
    if config.verbose:
        print("=" * 100)
    headers = request.headers
    wandb_api_key = headers.get("wandb-api-key")
    if wandb_api_key is None:
        return JSONResponse(status_code=400, content={"error": "wandb-api-key header is required"})
    logger.info(f"Headers: {headers}")
    # Get the request body
    body = await request.body()
    # Parse the JSON body
    try:
        data = json.loads(body)
        data["wandb_api_key"] = wandb_api_key
        payload = IllustratePayload.model_validate(data)
        if config.verbose:
            print("=" * 100)
            print(f"Received payload: {payload}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid Payload: {e}"})

    # Get the user's IP address
    client_ip = request.client.host
    num_images = len(payload.story.split("\n\n"))
    logger.info(f"Got request from client IP: {client_ip} to generate {num_images} images")
    if config.verbose:
        print("=" * 100)

    try:
        # Use ProcessPoolExecutor to run generate_illustration in a separate process
        
        # Wait for the result
        async with semaphore:  # Use semaphore to limit concurrent requests
            stdout, stderr = await generate_illustration(payload)
            if config.verbose:
                print("="*100)
                print(stdout[:300])
                print("...")
                print("="*100)
            # Record the request and update stats
            stats.record_request(time.time(), 0, client_ip)  # Update with actual token count if available
            image_base64_dict = get_images_dict(stdout)
            
            # compute some stats
            current_time = time.time()
            total_images = len(image_base64_dict)

            stats.record_request(current_time, total_images, client_ip)

            return JSONResponse(content={
                "result": f"Generated {total_images} images successfully",
                "weave_trace": get_weave_link(stdout),
                "images": image_base64_dict,
            })
    except Exception as e:
        logger.error(f"Error generating illustration: {e}")
        if "wandb login" in str(e):
            return JSONResponse(status_code=500, content={"error": "wandb login failed, please check your wandb api key"})
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')  # Use 'spawn' method for better compatibility
    uvicorn.run(app, host=config.server_host, port=config.server_port)