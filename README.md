# Mini RAG Document Assistant

A small, modular Retrieval-Augmented Generation (RAG) app using LangChain, Streamlit, ChromaDB, Hugging Face embeddings, and Groq.

## Features
- Upload text-based PDFs and split them into overlapping chunks.
- Embed and persist chunks in ChromaDB.
- Retrieve top-K similar chunks for a question.
- Generate context-grounded answers through Groq.
- Display source filename, page number, retrieved passages, and vector distance.
- Evaluate source retrieval with Hit Rate@K, a source-level Recall@K proxy, Precision@K, and MRR.

## Requirements
- Python 3.10 or 3.11 recommended.
- A Groq API key: https://console.groq.com/
- Internet access for installing dependencies, downloading the embedding model, and calling Groq.

## Windows / VS Code setup
Open a terminal in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` to your real key. Never commit `.env`.

Run:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually http://localhost:8501.

## Add PDFs
1. Open the app.
2. Use the sidebar to select one or more text-based PDFs.
3. Click **Index uploaded PDFs** and wait for indexing to finish.
4. Ask questions in the main panel. Expand retrieved sources to inspect evidence.

PDFs and Chroma data are stored under `data/`, which is excluded from Git. Scanned/image-only PDFs are not OCR'd by this prototype.

## Evaluation metrics
1. Index PDFs in the app first.
2. Edit `evaluation_questions.json`.
3. For each question, write `expected_sources` with exact uploaded filenames that genuinely contain the answer/evidence. Replace `sample_report.pdf`.
4. Run in a second terminal with the same virtual environment activated:

```powershell
python evaluate.py
```

- **Hit Rate@K**: fraction of questions where at least one expected source file is in the top K.
- **Recall@K proxy**: equals Hit Rate@K in this file-level evaluation; it is not true chunk-level recall.
- **Precision@K**: fraction of top-K retrieved chunks that come from expected source files, averaged across questions.
- **MRR**: mean reciprocal rank of the first retrieved chunk from an expected source file.

These are retrieval metrics, not answer correctness or faithfulness metrics. Use a manually checked dataset and do not claim scores until you run the evaluation. For generation evaluation, consider Ragas with reference answers and relevant contexts.

## Troubleshooting
- `GROQ_API_KEY is missing`: check `.env` is in the root project folder and the key is valid.
- Model unavailable: set `GROQ_MODEL` to a model currently available to your Groq account.
- No text extracted: the PDF may be scanned; OCR is not included.
- First run may take a while because the embedding model is downloaded.
- If a package API changes, recreate the virtual environment and install current compatible versions.
