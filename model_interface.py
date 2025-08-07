from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ModelInterface(ABC):
    """Abstract interface for all model implementations."""
    
    @abstractmethod
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer from question and context."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    def get_embedding_object(self):
        """Get the embedding object for Chroma."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate model configuration."""
        pass 