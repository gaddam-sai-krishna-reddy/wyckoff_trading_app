import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

def get_optimal_device():
    """Detect and return the optimal device for model inference."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        print(f"GPU detected: {gpu_name}")
        print(f"Number of GPUs: {gpu_count}")
        print("Using CUDA for faster embedding generation.")
        return "cuda"
    else:
        print("GPU not available. Using CPU for embedding generation.")
        return "cpu"

def get_collection_name():
    """Get collection name based on model type."""
    model_type = os.getenv("MODEL_TYPE", "google_flan")
    base_name = "wyckoff_qa"
    if model_type == "chatgpt":
        return f"gpt_{base_name}"
    elif model_type == "google_flan":
        return f"flan_{base_name}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_embedding_model():
    """Get appropriate embedding model based on model type."""
    model_type = os.getenv("MODEL_TYPE", "google_flan")
    optimal_device = get_optimal_device()
    
    if model_type == "chatgpt":
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for ChatGPT model")
            print("Using OpenAI text-embedding-3-small for ChatGPT model")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_api_key
            )
        except ImportError:
            print("Warning: OpenAI embeddings not available, falling back to HuggingFace")
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", 
                model_kwargs={"device": optimal_device}
            )
    elif model_type == "google_flan":
        print("Using HuggingFace all-MiniLM-L6-v2 for Google Flan model")
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={"device": optimal_device}
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def build_and_persist(csv_path: str, persist_dir: str = "chroma_db"):
    """
    Load Q&A CSV, chunk, embed with model-specific embeddings, and persist to Chroma on disk.
    """
    # Get model-specific collection name and embedding model
    collection_name = get_collection_name()
    embedding_model = get_embedding_model()
    
    print(f"Building index for model type: {os.getenv('MODEL_TYPE', 'google_flan')}")
    print(f"Collection name: {collection_name}")
    print(f"Embedding model: {type(embedding_model).__name__}")
    
    # 1) Load CSV
    df = pd.read_csv(csv_path)

    # 2) Create Documents with BETTER structure for RAG
    docs = []
    for _, row in df.iterrows():
        question = row["Questions"]
        answer = row["Answers"]
        
        # Create document with both Q&A in content for better retrieval
        content = f"Question: {question}\nAnswer: {answer}"
        
        doc = Document(
            page_content=content,
            metadata={
                "question": question,
                "answer": answer,
                "type": "qa_pair"
            }
        )
        docs.append(doc)

    # 3) Chunk with better parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for better context
        chunk_overlap=200,  # More overlap for continuity
        separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting
    )
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks from {len(docs)} documents")

    # 4) Build persistent Chroma index with model-specific embeddings
    print(f"Building embeddings with {type(embedding_model).__name__}")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    # 5) Persist to disk
    vectorstore.persist()
    print(f"Indexing complete. Persisted to '{persist_dir}' as '{collection_name}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and persist QA index from CSV into Chroma DB")
    parser.add_argument("--csv_path", default="data.csv", help="Path to your QA CSV file")
    parser.add_argument("--persist_dir", default="chroma_db", help="Directory for Chroma to persist the DB")
    args = parser.parse_args()
    build_and_persist(args.csv_path, args.persist_dir)