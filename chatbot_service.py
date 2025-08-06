from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import torch
import logging
from typing import Optional, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Configuration class for model settings."""
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GENERATION_MODEL = "google/flan-t5-large"
    CHROMA_PERSIST_DIR = "chroma_db"
    COLLECTION_NAME = "wyckoff_qa"
    MAX_LENGTH = 128
    SEARCH_K = 3
    
    @staticmethod
    def get_model_kwargs(device: str) -> Dict[str, Any]:
        """Get model kwargs based on device."""
        return {"device": device}

# Enhanced GPU detection and optimization with error handling
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

def initialize_embeddings(device: str) -> HuggingFaceEmbeddings:
    """Initialize embeddings with error handling."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs=Config.get_model_kwargs(device)
        )
        logger.info(f"Embeddings initialized successfully on {device}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise

def initialize_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Initialize vectorstore with error handling."""
    try:
        vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION_NAME
        )
        logger.info("Vectorstore initialized successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore: {e}")
        raise

def initialize_text_generator(device: str) -> HuggingFacePipeline:
    """Initialize text generator with error handling."""
    try:
        device_id = 0 if device == "cuda" else -1
        logger.info(f"Text generation device: {device_id} ({'GPU' if device_id >= 0 else 'CPU'})")
        
        text_gen = pipeline(
            "text2text-generation",
            model=Config.GENERATION_MODEL,
            device=device_id,
            max_length=Config.MAX_LENGTH,
            do_sample=False,
        )
        generator_llm = HuggingFacePipeline(pipeline=text_gen)
        logger.info("Text generator initialized successfully")
        return generator_llm
    except Exception as e:
        logger.error(f"Failed to initialize text generator: {e}")
        raise

# Initialize components with error handling
def initialize_components():
    """Initialize all components with proper error handling."""
    try:
        # Get optimal device
        optimal_device = get_optimal_device()
        logger.info(f"Selected device: {optimal_device}")
        
        # Initialize embeddings
        embeddings = initialize_embeddings(optimal_device)
        
        # Initialize vectorstore
        vectorstore = initialize_vectorstore(embeddings)
        
        # Initialize text generator
        generator_llm = initialize_text_generator(optimal_device)
        
        return vectorstore, generator_llm
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Initialize components
vectorstore, generator_llm = initialize_components()

# RAG chain with error handling
prompt_rag = ChatPromptTemplate.from_template(
    """You are an expert trading assistant specializing in Wyckoff methodology. 

Based on the following context, provide a helpful and accurate answer to the user's question. Use the information from the provided context to give the best possible answer. Be specific and detailed in your response.

Context:
{context}

Question: {question}

Answer:"""
)

# Enhanced retrieval with better parameters
def get_enhanced_retriever(vectorstore, search_k: int = None):
    """Create an enhanced retriever with better search parameters."""
    # Use provided search_k or default to Config.SEARCH_K
    k_value = search_k if search_k is not None else Config.SEARCH_K
    
    return vectorstore.as_retriever(
        search_type="similarity",  # or "mmr" for diversity
        search_kwargs={
            "k": k_value * 2,  # Get more candidates for better filtering (2x the final count)
        }
    )



def retrieve_docs(question: str, search_k: int = None) -> list:
    """Retrieve documents with error handling and re-ranking."""
    try:
        # Use provided search_k or default to Config.SEARCH_K
        k_value = search_k if search_k is not None else Config.SEARCH_K
        
        # Create dynamic retriever based on search_k
        dynamic_retriever = get_enhanced_retriever(vectorstore, search_k)
        
        # Get initial candidates
        docs = dynamic_retriever.get_relevant_documents(question)
        
        # Simple re-ranking: prioritize docs with both Q&A and filter by relevance
        scored_docs = []
        for doc in docs:
            score = 0
            content = doc.page_content.lower()
            question_lower = question.lower()
            
            # Boost score for exact question matches
            if question_lower in content:
                score += 3
            # Boost score for keyword matches
            keywords = question_lower.split()
            for keyword in keywords:
                if len(keyword) > 3 and keyword in content:
                    score += 1
            
            # Additional scoring based on content quality
            if "question:" in content and "answer:" in content:
                score += 1  # Boost for well-structured Q&A pairs
            
            # Only include docs with minimum relevance score
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top k_value documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_docs[:k_value]]
        
        logger.info(f"Retrieved {len(top_docs)} documents for question: {question[:50]}...")
        return top_docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def get_answer(question: str, search_k: int = None) -> str:
    """Get answer with enhanced RAG processing."""
    try:
        # Get relevant documents
        docs = retrieve_docs(question, search_k)
        
        if not docs:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing or ask a different question about Wyckoff trading methodology."
        
        # Create context from documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Source {i}:\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with enhanced prompt
        answer = final_rag_chain.invoke({
            "context": context,
            "question": question
        })
        
        logger.info(f"Generated answer for question: {question[:50]}...")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Sorry, I encountered an error while processing your question: {str(e)}"

# Enhanced RAG chain
final_rag_chain = (
    prompt_rag
    | generator_llm
    | StrOutputParser()
)
