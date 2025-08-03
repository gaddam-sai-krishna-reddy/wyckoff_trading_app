from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.load import dumps, loads
from operator import itemgetter

# 1) Build your retriever as a RunnableSequence
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="wyckoff_qa"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2) Generator LLM
text_gen = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=0,
    max_length=128,
    do_sample=False,
)
generator_llm = HuggingFacePipeline(pipeline=text_gen)

# 3) Multi-query paraphrasing Runnable

multi_query_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(
    multi_query_template
)
generate_queries = (
    prompt_perspectives
    | generator_llm
    | StrOutputParser()
    | (lambda out: out.split("\n"))
)

# 4) Retrieval-chain mirroring your notebook
def retrieval_chain(question: str):
    # 4a) split into alternative questions
    alts = generate_queries.invoke({"question": question})
    # 4b) map each alt -> docs list
    docs_lists = [retriever.get_relevant_documents(q) for q in alts]
    # 4c) dedupe
    flat = [dumps(doc) for sub in docs_lists for doc in sub]
    unique = set(flat)
    return [loads(s) for s in unique]

# 5) Final RAG chain
prompt_rag = ChatPromptTemplate.from_template(
    "Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"
)
final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | prompt_rag
    | generator_llm
    | StrOutputParser()
)
