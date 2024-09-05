from dataclasses import dataclass

@dataclass
class Config:
    # Logging
    log_level: str = "INFO"

    # Rate Limit
    rate_limit_requests_per_minute: int = 5
    rate_limit_error_message: str = "Rate limit exceeded. Please try again later."

    # Model
    model_api_endpoint1: str = "http://195.242.24.224:8000/v1"
    model_api_key1: str = "dummy-key"
    model_name1: str = "mistralai/Mistral-Large-Instruct-2407"
    model_api_endpoint2: str = "http://195.242.24.252:8000/v1"
    model_api_key2: str = "dummy-key"
    model_name2: str = "mistralai/Mistral-Nemo-Instruct-2407"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000

config = Config()