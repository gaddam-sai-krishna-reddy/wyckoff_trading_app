from typing import Dict, Any
from model_interface import ModelInterface
from google_flan_model import GoogleFlanModel
from chatgpt_model import ChatGPTModel

class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> ModelInterface:
        """Create model instance based on type."""
        if model_type == "google_flan":
            return GoogleFlanModel(config)
        elif model_type == "chatgpt":
            return ChatGPTModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type."""
        if model_type == "google_flan":
            return {
                "max_length": 128,
                "do_sample": False,
                "temperature": 0.0
            }
        elif model_type == "chatgpt":
            return {
                "temperature": 0.1,
                "max_tokens": 1000,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "openai_api_key": None  # Must be provided via environment
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_supported_models() -> list:
        """Get list of supported model types."""
        return ["google_flan", "chatgpt"] 