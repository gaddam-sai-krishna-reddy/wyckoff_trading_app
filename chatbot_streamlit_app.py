# streamlit_app.py
import streamlit as st
#uncomment for multiquery
# from chatbot_service import retrieval_chain, final_rag_chain
from chatbot_service import retrieve_docs,final_rag_chain

def main():
    st.title("📚 Q&A Chatbot")
    q = st.text_input("Ask a question:")
    if st.button("Submit") and q:
        #uncomment for multiquery
        # docs = retrieval_chain(q)
        # context = "\n\n".join(d.page_content for d in docs)
        # answer = final_rag_chain.invoke({"question": q, "context": context})
        answer = final_rag_chain.invoke(q)
        docs = retrieve_docs(q)
        print("Answer:", answer)
        st.subheader("Answer:")
        st.write(answer)
        with st.expander("Source Chunks"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}:** {d.page_content}")

if __name__ == "__main__":
    main()
