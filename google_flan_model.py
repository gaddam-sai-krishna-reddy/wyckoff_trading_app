from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import logging
from typing import Dict, Any, List
from model_interface import ModelInterface

logger = logging.getLogger(__name__)

def get_optimal_device() -> str:
    """Detect and return the optimal device for model inference with error handling."""
    try:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"GPU detected: {gpu_name}")
            logger.info(f"Number of GPUs: {gpu_count}")
            logger.info("Using CUDA for faster inference.")
            return "cuda"
        else:
            logger.info("GPU not available. Using CPU.")
            return "cpu"
    except Exception as e:
        logger.warning(f"Error during device detection: {e}. Falling back to CPU.")
        return "cpu"

class GoogleFlanModel(ModelInterface):
    """Google Flan-T5 model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = "google/flan-t5-large"
        self.device = get_optimal_device()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Flan-T5 model."""
        try:
            self.text_gen = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                max_length=self.config.get("max_length", 128),
                do_sample=self.config.get("do_sample", False)
            )
            self.generator = HuggingFacePipeline(pipeline=self.text_gen)
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": self.device}
            )
            logger.info(f"Google Flan model initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Flan model: {e}")
            raise
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Flan-T5."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """You are an expert trading assistant specializing in Wyckoff methodology. 

Based on the following context, provide a helpful and accurate answer to the user's question. Use the information from the provided context to give the best possible answer. Be specific and detailed in your response.

Context:
{context}

Question: {question}

Answer:"""
            )
            
            chain = prompt | self.generator | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            return answer
        except Exception as e:
            logger.error(f"Error generating answer with Flan: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using HuggingFace model."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error getting embeddings with Flan: {e}")
            raise
    
    def get_embedding_object(self):
        """Get the embedding object for Chroma."""
        return self.embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Google Flan-T5",
            "model": self.model_name,
            "device": self.device,
            "max_length": self.config.get("max_length", 128),
            "capabilities": ["text-generation", "embeddings"]
        }
    
    def validate_config(self) -> bool:
        """Validate model configuration."""
        required_keys = ["max_length", "do_sample"]
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys for Google Flan model: {missing_keys}")
        
        return True 