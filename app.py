import streamlit as st

from ingestion import ingest_pdf, indexed_chunk_count
from generation import answer_question

st.set_page_config(page_title="Mini RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Mini RAG Document Assistant")
st.write("Upload PDFs, ask questions, and inspect the retrieved evidence.")

with st.sidebar:
    st.header("1. Add documents")
    uploads = st.file_uploader(
        "Choose text-based PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded PDFs", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one PDF first.")
        else:
            for uploaded in uploads:
                try:
                    with st.spinner(f"Indexing {uploaded.name}..."):
                        count = ingest_pdf(uploaded)
                    st.success(f"{uploaded.name}: indexed {count} chunks")
                except Exception as exc:
                    st.error(f"{uploaded.name}: {exc}")

    try:
        st.metric("Indexed chunks", indexed_chunk_count())
    except Exception:
        st.caption("Vector database will initialize when first used.")

st.header("2. Ask a question")
question = st.text_input(
    "Question about your PDFs",
    placeholder="What are the main findings in this document?",
)
top_k = st.slider("Retrieved chunks (top K)", 1, 10, 4)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        try:
            with st.spinner("Retrieving passages and generating an answer..."):
                result = answer_question(question, top_k)
            st.subheader("Answer")
            st.markdown(result["answer"])
            st.subheader("Retrieved sources")
            if not result["sources"]:
                st.info("No source chunks were retrieved.")
            for i, source in enumerate(result["sources"], start=1):
                with st.expander(
                    f"Source {i} · {source['source']} · page {source['page']} "
                    f"· distance {source['distance']:.4f}"
                ):
                    st.write(source["text"])
        except Exception as exc:
            st.error(f"Request failed: {exc}")

st.divider()
st.caption("Prototype: scanned PDFs need OCR; evaluation metrics are estimates based on your labeled test set.")
