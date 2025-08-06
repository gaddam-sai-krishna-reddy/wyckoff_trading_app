# streamlit_app.py
import streamlit as st
import logging
from typing import Optional
import time

# Import from chatbot service
try:
    from chatbot_service import retrieve_docs, get_answer, logger
except ImportError as e:
    # Fallback if logger is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import logger from chatbot_service: {e}")
    from chatbot_service import retrieve_docs, get_answer

# Configure page
st.set_page_config(
    page_title="Wyckoff Trading Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .loading {
        text-align: center;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Wyckoff Trading Q&A Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar for additional features
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Search parameters
        st.subheader("Search Parameters")
        search_k = st.slider("Number of source chunks", min_value=1, max_value=10, value=3)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Question input
        question = st.text_input(
            "Ask a question about Wyckoff trading:",
            placeholder="e.g., What is a Spring in Wyckoff methodology?",
            key="question_input"
        )
        
        # Submit button
        submit_button = st.button("Submit", type="primary")
        
        # Process question
        if submit_button and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get answer
                    start_time = time.time()
                    answer = get_answer(question, search_k)
                    processing_time = time.time() - start_time
                    
                    # Get source documents
                    docs = retrieve_docs(question, search_k)
                    
                    # Display answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.subheader("💡 Answer:")
                    st.write(answer)
                    st.caption(f"⏱️ Processing time: {processing_time:.2f} seconds")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display source chunks
                    if docs:
                        with st.expander(f"📖 Source Chunks ({len(docs)} found)"):
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                                st.markdown(f"**Chunk {i}:** {doc.page_content}")
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.caption(f"Metadata: {doc.metadata}")
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No relevant source chunks found.")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": len(docs),
                        "time": processing_time
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error processing your question: {str(e)}")
                    logger.error(f"Streamlit app error: {e}")
    
    with col2:
        # Chat history
        st.subheader("💬 Recent Questions")
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"Q{i}: {chat['question'][:30]}..."):
                    st.write(f"**Q:** {chat['question']}")
                    st.write(f"**A:** {chat['answer']}")
                    st.caption(f"Sources: {chat['sources']} | Time: {chat['time']:.2f}s")
        else:
            st.info("No questions asked yet.")

if __name__ == "__main__":
    main()
