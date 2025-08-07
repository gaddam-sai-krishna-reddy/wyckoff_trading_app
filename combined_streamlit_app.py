# combined_streamlit_app.py
import streamlit as st
import logging
from typing import Optional
import time
import pandas as pd

# Import from chatbot service
try:
    from chatbot_service import retrieve_docs, get_answer, logger
except ImportError as e:
    # Fallback if logger is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import logger from chatbot_service: {e}")
    from chatbot_service import retrieve_docs, get_answer

# Import from wyckoff
from wyckoff import run_backtest, get_available_tickers

# Configure page
st.set_page_config(
    page_title="Wyckoff Trading Platform",
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
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
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
    .divider {
        margin: 3rem 0;
        border-top: 3px solid #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)

def chatbot_section():
    """Chatbot section at the top."""
    st.markdown('<h1 class="section-header">🤖 Wyckoff Trading Q&A Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar for chatbot settings
    with st.sidebar:
        st.header("⚙️ Chatbot Settings")
        
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

def backtest_section():
    """Backtest section at the bottom."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="section-header">📊 Wyckoff Strategy Backtest</h1>', unsafe_allow_html=True)
    
    # Sidebar for backtest settings
    with st.sidebar:
        st.header("📊 Backtest Settings")
        tickers = get_available_tickers()
        selected = st.selectbox("Choose a stock:", tickers)
        
        start_date = st.date_input(
            "Start date", value=pd.to_datetime("2020-01-01")
        )
        end_date = st.date_input(
            "End date", value=pd.to_datetime("2025-06-30")
        )
        
        run_backtest_button = st.button("▶️ Run Backtest")
    
    # Main backtest interface
    if run_backtest_button:
        with st.spinner(f"Running backtest on {selected}…"):
            try:
                equity_df, metrics = run_backtest(
                    selected,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )

                # Equity curve plot
                st.subheader(f"Wyckoff Strategy vs Buy-and-Hold — {selected}")
                st.line_chart(equity_df)

                # Metrics table
                st.subheader("Aggregate Results")
                df_metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
                st.table(df_metrics)
                
            except Exception as e:
                st.error(f"❌ Error running backtest: {str(e)}")
                logger.error(f"Backtest error: {e}")
    else:
        st.info("Select parameters and click 'Run Backtest' to start the analysis.")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Wyckoff Trading Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Your comprehensive platform for Wyckoff trading analysis and Q&A**")
    
    # Chatbot section
    chatbot_section()
    
    # Backtest section
    backtest_section()

if __name__ == "__main__":
    main() 