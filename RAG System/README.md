# Healthcare Information Assistant (RAG)

An AI-powered Retrieval-Augmented Generation (RAG) system for healthcare queries.

## Features
- **Semantic Search**: Retrieval of top-k documents based on medical query.
- **Accuracy Layer**: Generates responses ONLY from retrieved context.
- **Safety**: Emergency query detection and mandatory medical disclaimers.
- **Citations**: Source tracking in every response.

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set OpenAI API Key in `.env`:
    ```env
    OPENAI_API_KEY=your_key_here
    ```
3.  Run the application:
    ```bash
    streamlit run app.py
    ```

## Architecture
See `architecture_diagram.txt` for the technical flow.
