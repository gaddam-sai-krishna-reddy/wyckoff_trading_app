import logging
from typing import Dict, Any, List
from model_interface import ModelInterface
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class ChatGPTModel(ModelInterface):
    """OpenAI GPT model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = "gpt-4o"  # Updated model name
        self.api_key = config.get("openai_api_key")
        self._validate_api_key()
        self._initialize_openai()
    
    def _validate_api_key(self):
        """Validate OpenAI API key."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using GPT."""
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._create_user_prompt(question, context)}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self._get_generation_config()
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer with GPT: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error getting embeddings with GPT: {e}")
            raise
    
    def get_embedding_object(self):
        """Get the embedding object for Chroma."""
        return self.embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI GPT-4o",
            "model": self.model_name,
            "capabilities": ["text-generation", "embeddings"],
            "config": self._get_generation_config()
        }
    
    def validate_config(self) -> bool:
        """Validate model configuration."""
        required_keys = ["temperature", "max_tokens", "top_p", "openai_api_key"]
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys for ChatGPT model: {missing_keys}")
        
        return True
    
    def _get_system_prompt(self) -> str:
        return "You are an expert Wyckoff trading analyst. Answer the user's question based on the provided context."
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        return f"""**Instructions:**
- Provide the most direct and specific answer
- If multiple sources conflict, prioritize the most relevant one
- Be concise and focused

Context:
{context}

Question: {question}"""
    
    def _get_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 1000),
            "top_p": self.config.get("top_p", 0.9),
            "frequency_penalty": self.config.get("frequency_penalty", 0.1),
            "presence_penalty": self.config.get("presence_penalty", 0.1)
        } 