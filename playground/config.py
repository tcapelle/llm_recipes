from dataclasses import dataclass

@dataclass
class Config:
    # Logging
    log_level: str = "INFO"

    # Rate Limit
    rate_limit_requests_per_minute: int = 5
    rate_limit_error_message: str = "Rate limit exceeded. Please try again later."

    # Model
    model_api_endpoint: str = "http://0.0.0.0:8000/v1"
    model_api_key: str = "dummy-key"
    model_name: str = "mistralai/Mistral-Large-Instruct-2407"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8010

config = Config(model_name="mistralai/Mistral-Large-Instruct-2407")