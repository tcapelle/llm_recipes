import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import os
import json
import httpx
import logging
import openai
import wandb
from dataclasses import dataclass
import time
from collections import deque, Counter
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute

import simple_parsing
from pydantic import BaseModel, Field, field_validator

client = openai.AsyncOpenAI()
app = FastAPI()

from story_illustrator.models import FalAITextToImageGenerationModel, StoryIllustrator

def illustrate(
    story: str,
    story_title: str = "Gift of the Magi",
    story_author: str = "O. Henry",
    story_setting: str = "the year 1905, New York City, United States of America",
    openai_model: str = "gpt-4o",
    llm_seed: Optional[int] = None,
    txt2img_model_address: str = "fal-ai/flux-pro",
    image_size: str = "square",
    illustration_style: Optional[str] = None,
    use_text_encoder_2: bool = False,
):
    story_illustrator = StoryIllustrator(
        openai_model=openai_model,
        llm_seed=llm_seed,
        text_to_image_model=FalAITextToImageGenerationModel(
            model_address=txt2img_model_address
        ),
    )
    paragraphs = story.split("\n\n")
    story_illustrator.predict(
        story=story,
        metadata={
            "title": story_title,
            "author": story_author,
            "setting": story_setting,
        },
        paragraphs=paragraphs[:10],
        illustration_style=illustration_style,
        use_text_encoder_2=use_text_encoder_2,
        image_size=image_size,
    )
    

class IllustratePayload(BaseModel):
    story: str = Field(..., description="The text of the story to illustrate")
    story_title: str = Field(..., description="The title of the story")
    story_author: str = Field(..., description="The author of the story")
    story_setting: str = Field(..., description="The setting of the story")
    wandb_api_key: str = Field(..., description="The wandb api key", min_length=40, max_length=40)

    @field_validator('wandb_api_key')
    def validate_wandb_api_key(cls, v):
        if len(v) != 40:
            raise ValueError('wandb_api_key must be exactly 40 characters long')
        return v

def clean_wandb_api_key():
    if "WANDB_API_KEY" in os.environ:
        os.unsetenv('WANDB_API_KEY')
        del os.environ["WANDB_API_KEY"]
    
    # Delete the ~/.netrc file if it exists
    netrc_path = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc_path):
        try:
            os.remove(netrc_path)
            logger.info(f"Deleted {netrc_path}")
        except OSError as e:
            logger.error(f"Error deleting {netrc_path}: {e}")
    else:
        logger.info(f"{netrc_path} does not exist")

def generate_illustration_process(payload: IllustratePayload):
    # Set the WANDB_API_KEY environment variable
    clean_wandb_api_key()

    logger.info(f"Logging into wandb with key: {payload.wandb_api_key}")
    wandb.login(key=payload.wandb_api_key, relogin=True)
    
    import weave
    weave.init("illustration-project")
    result = illustrate(
        story=payload.story,
        story_title=payload.story_title,
        story_author=payload.story_author,
        story_setting=payload.story_setting,
    )
    clean_wandb_api_key()
    return result

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

config = simple_parsing.parse(Config)

# Create a ProcessPoolExecutor
process_pool = ProcessPoolExecutor(max_workers=config.max_concurrent_requests)

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

@app.post("/illustrate/{path:path}")
async def forward_request(request: Request, path: str):
    """
    Generate an illustration for a story.
    """
    print("=" * 100)
    headers = request.headers
    wandb_api_key = headers.get("wandb-api-key")
    if wandb_api_key is None:
        return JSONResponse(status_code=400, content={"error": "wandb-api-key header is required"})
    logger.info(f"Headers: {headers}")
    # Get the request body
    body = await request.body()
    logger.info(f"Body: {body}")

    # Parse the JSON body
    try:
        data = json.loads(body)
        data["wandb_api_key"] = wandb_api_key
        payload = IllustratePayload.model_validate(data)
        print("=" * 100)
        logger.info(f"Received payload: {payload}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid Payload: {e}"})

    # Get the user's IP address
    client_ip = request.client.host
    logger.info(f"Client IP: {client_ip}")
    print("=" * 100)

    try:
        # Use ProcessPoolExecutor to run generate_illustration in a separate process
        future = process_pool.submit(generate_illustration_process, payload)
        
        # Wait for the result
        result = await asyncio.get_event_loop().run_in_executor(None, future.result)
        
        # Record the request and update stats
        stats.record_request(time.time(), 0, client_ip)  # Update with actual token count if available
        
        # Check if the result is an image
        if isinstance(result, bytes):
            return StreamingResponse(io.BytesIO(result), media_type="image/png")
        else:
            return JSONResponse(content={"result": result})
    except Exception as e:
        logger.error(f"Error generating illustration: {e}")
        if "wandb login" in str(e):
            return JSONResponse(status_code=500, content={"error": "wandb login failed, please check your wandb api key"})
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')  # Use 'spawn' method for better compatibility
    uvicorn.run(app, host=config.server_host, port=config.server_port)