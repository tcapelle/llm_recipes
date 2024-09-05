from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import json
from openai import AsyncOpenAI
from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute
import logging
from config import config

app = FastAPI()

# Set up logging
logger = logging.getLogger('uvicorn.error')
logging.basicConfig(level=config.log_level)

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

model_client = AsyncOpenAI(
    base_url=config.model_api_endpoint,
    api_key=config.model_api_key,
)

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
    
    # Log the intercepted request
    logger.info(f"Forwarding request to /{path}")
    
    # Forward the request to the model
    try:
        data['model'] = config.model_name  # Set the model name from config
        response = await model_client.chat.completions.create(**data)
        return JSONResponse(content=response.dict())
    except Exception as e:
        logger.error(f"Error calling model: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    uvicorn.run(app, host=config.server_host, port=config.server_port)