### 📘 Semantic Book Recommender with LLMs

## 🖥️ Dashboard Preview
Here’s a preview of the interactive Gradio dashboard:

dashboard.png

This project develops a semantic book recommender system leveraging the power of Large Language Models (LLMs).
It allows users to input a book description and receive recommendations for similar books, classified by categories and emotional tones.

Project Components

- **Text Data Cleaning**: Preparing raw text data for effective use in LLMs.  
- **Semantic Search with Vector Database**: Using embeddings & ChromaDB for book similarity search.  
- **Zero-Shot Classification**: Categorizing books without labeled data.  
- **Emotion & Sentiment Analysis**: Extracting emotions such as joy, sadness, fear, etc.  
- **Interactive Dashboard with Gradio**: A creative and user-friendly web interface. 

🛠️ Tech Stack

Python

LangChain (for LLM orchestration)

Hugging Face Transformers (embeddings, classification, sentiment analysis)

ChromaDB (vector database)

Gradio (web interface)

PyTorch (for GPU acceleration if available)

▶️ Usage

Run the Gradio dashboard:

`python gradio-dashboard.py`

Access the app at http://127.0.0.1:7860/.
