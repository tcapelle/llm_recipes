import asyncio
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

import json
import logging
import os
import openai
import sys
from pathlib import Path
from dataclasses import dataclass
import time
from collections import deque, Counter
from rich.logging import RichHandler

from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute

import simple_parsing
import tempfile  # New import

client = openai.AsyncOpenAI()


@dataclass
class Config:
    server_url: str
    server_api_key: str
    rate_limit_requests_per_minute: int = 100
    timeout: float = 30
    max_concurrent_requests: int = 10  # New parameter for shore
    rate_limit_error_message: str = "Rate limit exceeded. Please try again later."
    server_host: str = "0.0.0.0"
    server_port: int = 8032
    log_level: str = "INFO"
    stats_window_size: int = 3600  # 1 hour window for statistics
    verbose: bool = False
    python_executable: str = "llama32.py"
    model: str = "meta-llama"

config = simple_parsing.parse(Config)

# Set up logging
logger = logging.getLogger("interceptor")
logging.basicConfig(
    level=config.log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
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
        "code": "rate_limit_exceeded",
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

        tokens_last_minute = sum(
            count for t, count in self.token_counts if t > one_minute_ago
        )
        tokens_last_hour = sum(
            count for t, count in self.token_counts if t > one_hour_ago
        )

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
            logger.info("PERIODIC STATS:")
            logger.info(f"Current RPS: {rps:.2f}")
            logger.info(f"RPM: {rpm}")
            logger.info(f"Tokens/min: {tokens_per_minute}")
            logger.info(f"Tokens/hour: {tokens_per_hour}")
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


async def run_python(program: Path, *args, timeout: float = config.timeout):
    """
    Run a Python program with the given input file and output file.
    """
    if config.verbose:
        print(f"Executing {sys.executable} {program} with args: {args}\n")
        print("=" * 100)

    try:
        custom_env = {
            "SERVER_API_KEY": config.server_api_key,
            "SERVER_URL": config.server_url,
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            program,
            *args,
            env=custom_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Program execution timed out after {timeout} seconds")

        if process.returncode != 0:
            raise RuntimeError(f"Program execution failed: {stderr.decode()}")

        return stdout.decode(), stderr.decode()

    except Exception as e:
        raise RuntimeError(f"Error running Python program: {str(e)}")


async def call_llama32(payload_file: Path):
    # Run the python script
    args = [f"--payload_file={payload_file}"]
    if config.verbose:
        print(f"Running {config.python_executable} with args:")
        for arg in args:
            print(f"  {arg}")
        print("=" * 100)
    return await run_python(
        Path(config.python_executable),
        *args,
    )

def get_wandb_user(stdout: str):
    """Extracts the W&B user:
    s = "Logged in as Weights & Biases user: geekyrakshit."
    user = get_wandb_user(s)
    assert user == "geekyrakshit"
    """
    match = re.search(r"Logged in as Weights & Biases user: (.*)\.", stdout)
    if match:
        return match.group(1).strip()
    return None

def get_output_dict(stdout: str):
    output_json = json.loads(stdout.split("<response>")[-1].split("</response>")[-2])
    return output_json


@app.post("/v1/chat/{path:path}")
async def forward_request(request: Request, path: str):
    # Get the request body
    body = await request.body()
    headers = request.headers
    # print(f"Headers: {headers}")
    # print("=" * 100)
    # print(f"Received payload: {body}")

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as temp_file:
        # Write the headers
        headers = dict(request.headers)
        body = json.loads(body)
        body["model"] = config.model
        json_payload = {"headers": headers, "body": body}
        json.dump(json_payload, temp_file, indent=2)
        temp_file.flush()  # Ensure all data is written to disk

        payload_path = Path(temp_file.name)
        # print(f"Payload path: {payload_path}")

        try:
            # Call llama32 with the payload file
            current_time = time.time()
            stdout, stderr = await call_llama32(payload_path)
            # print(f"Stdout: {stdout}")
            # print("=" * 100)
            # print(f"Stderr: {stderr}")
            # print("=" * 100)

            # Parse the output using get_output_dict
            output = get_output_dict(stdout)
            response = JSONResponse(content=output)
            current_time = time.time()
            total_tokens = output.get("usage", {}).get("total_tokens", 0)
            user_ip = request.client.host
            wandb_user = get_wandb_user(stdout)  # Update with actual token count if available
            logger.info(
                f"Served {total_tokens} tokens to [cyan]{wandb_user}[/] at {user_ip}"
            )
            stats.record_request(current_time, total_tokens, user_ip)
            return response
        except Exception as e:
            logger.error(f"Error generating request: {e}")
            wandb_error = """1. Sign up to a free Weights & Biases account at https://www.wandb.ai \n2. Set WANDB_API_KEY to your API key from https://www.wandb.ai/authorize"""
            if "wandb login" in str(e):
                return JSONResponse(status_code=500, content={"error": wandb_error})
            return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host=config.server_host, port=config.server_port)
