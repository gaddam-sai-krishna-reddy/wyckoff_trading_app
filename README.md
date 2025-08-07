<<<<<<< HEAD
# Wyckoff Trading Chatbot

A sophisticated RAG-based chatbot for Wyckoff trading methodology using an interface-based architecture that supports both Google Flan-T5 and ChatGPT models.

## 🏗 **Architecture Overview**

The chatbot uses an interface-based architecture with the Factory pattern for easy model switching:
- **ModelInterface**: Abstract base class for all model implementations
- **GoogleFlanModel**: Local HuggingFace Flan-T5 implementation
- **ChatGPTModel**: OpenAI GPT-4o implementation
- **ModelFactory**: Factory pattern for creating model instances
- **ChatbotService**: Service layer using model interface

## 📁 **Project Structure**

```
wyckoff_trading_app/
├── .env                           # Environment variables
├── chatbot_service.py             # Main service layer
├── model_interface.py             # Abstract interface
├── google_flan_model.py           # Google Flan-T5 implementation
├── chatgpt_model.py              # ChatGPT implementation
├── model_factory.py               # Factory pattern
├── build_index.py                 # Index building script
├── chatbot_streamlit_app.py      # Streamlit application
├── requirements.txt               # Python dependencies
├── data.csv                      # Q&A dataset
└── README.md
```

## 🐍 **Environment Setup**

### **1. Create Anaconda Environment**

```bash
# Create a new conda environment
conda create -n wyckoff_chatbot python=3.9

# Activate the environment
conda activate wyckoff_chatbot
```

### **2. Install Dependencies**

```bash
# Install required packages
pip install -r requirements.txt
```

## 🔧 **Configuration**

### **1. Create `.env` file:**

```bash
# ===========================================
# COMMON ENVIRONMENT VARIABLES
# ===========================================

# Model Configuration
MODEL_TYPE=google_flan

# Database Settings
CHROMA_PERSIST_DIR=chroma_db

# Logging Configuration
LOG_LEVEL=INFO

# ===========================================
# GOOGLE FLAN-T5 SPECIFIC SETTINGS
# ===========================================

# Text Generation Parameters
TEMPERATURE=0.0
MAX_LENGTH=128
DO_SAMPLE=False

# ===========================================
# CHATGPT SPECIFIC SETTINGS (COMMENTED)
# ===========================================

# Uncomment and set these when using MODEL_TYPE=chatgpt
# OPENAI_API_KEY=your_openai_api_key_here
# TEMPERATURE=0.1
# MAX_TOKENS=1000
# TOP_P=0.9
# FREQUENCY_PENALTY=0.1
# PRESENCE_PENALTY=0.1
```

### **2. Collection Names**

The system automatically creates model-specific collection names:
- **Google Flan**: `flan_wyckoff_qa`
- **ChatGPT**: `gpt_wyckoff_qa`

This allows you to:
- Use different embedding models for each collection
- Keep data separate between models
- Switch models without affecting existing data
- Compare performance between different embeddings

## 🏗 **Building the Vector Database**

### **1. Index Creation**

The `build_index.py` script automatically:
- Reads your `data.csv` file
- Uses model-specific embeddings (HuggingFace for Flan, OpenAI for ChatGPT)
- Creates model-specific collections
- Persists to ChromaDB

### **2. Run Index Building**

```bash
# For Google Flan (default)
python build_index.py --csv_path data.csv

# For ChatGPT (set MODEL_TYPE=chatgpt in .env)
MODEL_TYPE=chatgpt python build_index.py --csv_path data.csv

# With custom persist directory
python build_index.py --csv_path data.csv --persist_dir custom_chroma_db
```

### **3. Command Line Options**

```bash
python build_index.py --help
```

**Available options:**
- `--csv_path`: Path to your CSV file (default: data.csv)
- `--persist_dir`: Directory for ChromaDB persistence (default: chroma_db)

## 🚀 **Running the Application**

### **1. Start the Streamlit App**

```bash
# Make sure your environment is activated
conda activate wyckoff_chatbot

# Run the application
streamlit run chatbot_streamlit_app.py
```

### **2. Access the Application**

Open your browser and navigate to:
```
http://localhost:8501
```

### **3. Usage**

1. **Ask Questions**: Type your Wyckoff-related questions
2. **Get Answers**: The chatbot will retrieve relevant documents and generate answers
3. **Switch Models**: Change `MODEL_TYPE` in `.env` to switch between models

## 🔄 **Model Switching**

### **Development Mode (Google Flan-T5):**
```bash
# .env
MODEL_TYPE=google_flan
```

### **Production Mode (ChatGPT):**
```bash
# .env
MODEL_TYPE=chatgpt
OPENAI_API_KEY=sk-your-actual-key
```

## 🛠 **Advanced Configuration**

### **1. Custom Parameters**

You can adjust model parameters in `.env`:

```bash
# For Google Flan
TEMPERATURE=0.0
MAX_LENGTH=128
DO_SAMPLE=False

# For ChatGPT
TEMPERATURE=0.1
MAX_TOKENS=1000
TOP_P=0.9
```

## 🎯 **Recommendation**

- **Development**: Use `MODEL_TYPE=google_flan`
- **Production**: Use `MODEL_TYPE=chatgpt` with proper API key
- **Testing**: Use both to compare performance

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.