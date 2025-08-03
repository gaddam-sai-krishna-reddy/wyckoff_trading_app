import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_and_persist(csv_path: str, persist_dir: str = "chroma_db", collection_name: str = "qa_index"):
    """
    Load Q&A CSV, chunk, embed with all-MiniLM-L6-v2, and persist to Chroma on disk.
    """
    # 1) Load CSV
    df = pd.read_csv(csv_path)

    # 2) Create Documents
    docs = [
        Document(
            page_content=row["Questions"],
            metadata={"answer": row["Answers"]}
        )
        for _, row in df.iterrows()
    ]

    # 3) Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # 4) Build persistent Chroma index
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}),
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
    parser.add_argument("--collection_name", default="wyckoff_qa", help="Logical name for this collection")
    args = parser.parse_args()
    build_and_persist(args.csv_path, args.persist_dir, args.collection_name)