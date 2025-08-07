from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from operator import itemgetter
import logging
from typing import Optional, Dict, Any, List
import os

# Import from separate model files
from model_interface import ModelInterface
from model_factory import ModelFactory

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Configure logging after loading .env
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

logger.info(f"Log level: {log_level}")

# Configuration
class Config:
    """Configuration class for model settings."""
    
    # Model type
    MODEL_TYPE = os.getenv("MODEL_TYPE", "google_flan")
    
    # Database settings
    CHROMA_PERSIST_DIR = "chroma_db"
    SEARCH_K = 3
    
    @staticmethod
    def get_collection_name() -> str:
        """Get collection name based on model type."""
        base_name = "wyckoff_qa"
        if Config.MODEL_TYPE == "chatgpt":
            return f"gpt_{base_name}"
        elif Config.MODEL_TYPE == "google_flan":
            return f"flan_{base_name}"
        else:
            raise ValueError(f"Unknown model type: {Config.MODEL_TYPE}")
    
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Get configuration for current model."""
        if Config.MODEL_TYPE == "chatgpt":
            return {
                "temperature": float(os.getenv("TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("MAX_TOKENS", "1000")),
                "top_p": float(os.getenv("TOP_P", "0.9")),
                "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY", "0.1")),
                "presence_penalty": float(os.getenv("PRESENCE_PENALTY", "0.1")),
                "openai_api_key": os.getenv("OPENAI_API_KEY")
            }
        elif Config.MODEL_TYPE == "google_flan":
            return {
                "max_length": int(os.getenv("MAX_LENGTH", "128")),
                "do_sample": os.getenv("DO_SAMPLE", "False").lower() == "true",
                "temperature": float(os.getenv("TEMPERATURE", "0.0"))
            }
        else:
            raise ValueError(f"Unknown model type: {Config.MODEL_TYPE}")
    
    @staticmethod
    def get_model_kwargs(device: str) -> Dict[str, Any]:
        """Get model kwargs based on device."""
        return {"device": device}

# Enhanced retrieval with similarity search
def get_enhanced_retriever(vectorstore, search_k: int = None):
    """Create an enhanced retriever with similarity search."""
    # Use provided search_k or default to Config.SEARCH_K
    k_value = search_k if search_k is not None else Config.SEARCH_K
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_value * 2} # Get more candidates for better filtering (2x the final count)
    )

def remove_duplicates(docs: list) -> list:
    """Remove duplicate documents from top k candidates."""
    unique_docs = []
    seen_content = set()
    
    for doc in docs:
        # Create simplified version for comparison
        simplified = doc.page_content.lower().replace(' ', '')[:100]  # First 100 chars
        
        if simplified not in seen_content:
            unique_docs.append(doc)
            seen_content.add(simplified)
    
    return unique_docs

def filter_by_specificity(content: str, question: str) -> bool:
    """Filter chunks based on how well they answer the specific question."""
    
    # Extract key terms from question
    question_lower = question.lower()
    question_terms = [word for word in question_lower.split() if len(word) > 3]
    
    # Extract key terms from content
    content_lower = content.lower()
    
    # Check for exact question match (highest priority)
    if question_lower in content_lower:
        return True
    
    # Check for key term matches
    matches = 0
    for term in question_terms:
        if term in content_lower:
            matches += 1
    
    # Calculate specificity score
    specificity_score = matches / len(question_terms) if question_terms else 0
    
    # Only keep chunks with good specificity
    if specificity_score < 0.4:
        return False
    
    # Additional check: if the content contains a Q&A pair, ensure the question is relevant
    # if "question:" in content_lower and "answer:" in content_lower:
    #     # Extract the question from the content
    #     try:
    #         qa_parts = content.split("Question:")
    #         if len(qa_parts) > 1:
    #             content_question = qa_parts[1].split("Answer:")[0].strip().lower()
    #             # Check if the content question is similar to the user question
    #             content_question_terms = [word for word in content_question.split() if len(word) > 3]
    #             question_overlap = sum(1 for term in question_terms if term in content_question_terms)
    #             question_relevance = question_overlap / len(question_terms) if question_terms else 0
    #             
    #             # Require moderate question relevance for Q&A pairs
    #             if question_relevance < 0.3:
    #                 return False
    #     except:
    #         pass
    
    return True

# ChatbotService class using the interface
class ChatbotService:
    """Service layer using model interface."""
    
    def __init__(self):
        self.model = self._initialize_model()
        self.vectorstore = self._initialize_vectorstore()
    
    def _initialize_model(self) -> ModelInterface:
        """Initialize model using factory."""
        config = Config.get_model_config()
        model = ModelFactory.create_model(Config.MODEL_TYPE, config)
        
        if not model.validate_config():
            raise ValueError(f"Invalid configuration for {Config.MODEL_TYPE}")
        
        logger.info(f"Model initialized: {model.get_model_info()}")
        return model
    
    def _initialize_vectorstore(self):
        """Initialize vectorstore with model embeddings."""
        return Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=self.model.get_embedding_object(),
            collection_name=Config.get_collection_name()
        )
    
    def retrieve_docs(self, question: str, search_k: int = None) -> list:
        """Retrieve documents with error handling and re-ranking."""
        try:
            # Use provided search_k or default to Config.SEARCH_K
            k_value = search_k if search_k is not None else Config.SEARCH_K
            
            # Create dynamic retriever based on search_k
            dynamic_retriever = get_enhanced_retriever(self.vectorstore, search_k)
            
            # Get initial candidates
            docs = dynamic_retriever.get_relevant_documents(question)
            
            # Enhanced re-ranking: prioritize docs with exact question matches and better Q&A relevance
            scored_docs = []
            for doc in docs:
                score = 0
                content = doc.page_content.lower()
                question_lower = question.lower()
                
                # Boost score for exact question matches (highest priority)
                if question_lower in content:
                    score += 10  # Increased from 3 to 10
                
                # Check for very similar questions in Q&A pairs
                if "question:" in content and "answer:" in content:
                    try:
                        qa_parts = doc.page_content.split("Question:")
                        if len(qa_parts) > 1:
                            content_question = qa_parts[1].split("Answer:")[0].strip().lower()
                            # Check for high similarity between questions
                            if content_question in question_lower or question_lower in content_question:
                                score += 8
                            elif any(word in content_question for word in question_lower.split() if len(word) > 4):
                                score += 5
                    except:
                        pass
                    
                    score += 1  # Boost for well-structured Q&A pairs
                
                # Boost score for keyword matches
                keywords = question_lower.split()
                for keyword in keywords:
                    if len(keyword) > 3 and keyword in content:
                        score += 1
                
                # Only include docs with minimum relevance score
                if score > 0:
                    scored_docs.append((score, doc))
            
            # Sort by score and return top k_value documents
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:k_value]]
            
            # Apply specificity filtering
            filtered_docs = []
            
            # Only calculate debug scores if debug logging is enabled
            debug_scores = []
            if logger.isEnabledFor(logging.DEBUG):
                for doc in top_docs:
                    # Calculate scores for debugging
                    question_lower = question.lower()
                    question_terms = [word for word in question_lower.split() if len(word) > 3]
                    content_lower = doc.page_content.lower()
                    
                    # Calculate specificity score
                    matches = 0
                    for term in question_terms:
                        if term in content_lower:
                            matches += 1
                    specificity_score = matches / len(question_terms) if question_terms else 0
                    
                    # Store debug info
                    debug_info = {
                        'specificity_score': specificity_score,
                        'passed_filter': filter_by_specificity(doc.page_content, question),
                        'content_preview': doc.page_content[:100] + "..."
                    }
                    debug_scores.append(debug_info)
            
            # Apply filtering to all documents
            for doc in top_docs:
                if filter_by_specificity(doc.page_content, question):
                    filtered_docs.append(doc)
            
            # Print debug information only at DEBUG level
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"=== SPECIFICITY FILTERING DEBUG ===")
                logger.debug(f"Question: {question}")
                logger.debug(f"Total docs before filtering: {len(top_docs)}")
                logger.debug(f"Docs after filtering: {len(filtered_docs)}")
                
                for i, debug_info in enumerate(debug_scores):
                    status = "✓ PASSED" if debug_info['passed_filter'] else "✗ FAILED"
                    logger.debug(f"Doc {i+1}: {status}")
                    logger.debug(f"  Specificity Score: {debug_info['specificity_score']:.3f}")
                    logger.debug(f"  Content: {debug_info['content_preview']}")
                    logger.debug("---")
            
            # If specificity filtering removed too many docs, include some of the original top docs
            if len(filtered_docs) < max(1, k_value // 2):
                docs_to_add = k_value - len(filtered_docs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Adding back {docs_to_add} docs due to aggressive filtering")
                
                # Add back only the top docs that didn't pass specificity, limited to docs_to_add
                docs_added = 0
                for doc in top_docs:
                    if doc not in filtered_docs and docs_added < docs_to_add:
                        filtered_docs.append(doc)
                        docs_added += 1
                    if docs_added >= docs_to_add:
                        break
            
            # Remove duplicates from filtered docs
            filtered_docs = remove_duplicates(filtered_docs)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Final retrieved documents: {len(filtered_docs)}")
                logger.debug("=== END DEBUG ===")
            
            return filtered_docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def get_answer(self, question: str, search_k: int = None) -> str:
        """Get answer using abstracted model."""
        try:
            # Get relevant documents
            docs = self.retrieve_docs(question, search_k)
            
            if not docs:
                return "I couldn't find any relevant information to answer your question. Please try rephrasing or ask a different question about Wyckoff trading methodology."
            
            # Create context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer using model interface
            answer = self.model.generate_answer(question, context)
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

# Initialize service for backward compatibility
def initialize_components():
    """Initialize all components with proper error handling."""
    try:
        service = ChatbotService()
        return service.vectorstore, service.model
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Initialize components
vectorstore, model = initialize_components()

# Backward compatibility functions
def retrieve_docs(question: str, search_k: int = None) -> list:
    """Backward compatibility function."""
    service = ChatbotService()
    return service.retrieve_docs(question, search_k)

def get_answer(question: str, search_k: int = None) -> str:
    """Backward compatibility function."""
    service = ChatbotService()
    return service.get_answer(question, search_k)
