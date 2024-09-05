from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import json
import httpx
import logging
import os
from dataclasses import dataclass

from limits import storage
from limits.strategies import FixedWindowRateLimiter
from limits import RateLimitItemPerMinute

app = FastAPI()

@dataclass
class Config:
    rate_limit_requests_per_minute: int = 20
    rate_limit_error_message: str = "Rate limit exceeded. Please try again later."
    mistral_api_endpoint: str = "https://api.mistral.ai/v1/chat/completions"
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY")
    request_timeout: int = 30
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "INFO"

config = Config()


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

    headers = request.headers
    # headers["authorization"] = f"Bearer {config.mistral_api_key}"
    print(f"Headers: {headers}")
    
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
        print(f"Headers: {json.dumps(headers, indent=4)}")
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        print(f"Data being sent: {json_data}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                data=json_data,
                headers=headers,
                timeout=config.request_timeout,
            )
        
        logger.info(f"Forwarding request to Mistral API")
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        logger.error(f"Error calling Mistral API: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    uvicorn.run(app, host=config.server_host, port=config.server_port)