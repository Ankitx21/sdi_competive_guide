import os
import re
import time
import warnings

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai


CHAT_MODEL = "gemini-2.5-flash"
INDEX_FILE = "doc_index.npz"
TOP_K = 5
MIN_SIM = 0.45


def retry_wait_seconds(err_text: str, default_wait=20):
    match = re.search(r"retry in ([0-9.]+)s", err_text, flags=re.IGNORECASE)
    if match:
        try:
            return max(default_wait, int(float(match.group(1))) + 1)
        except Exception:
            return default_wait
    return default_wait


def embed_query_with_retry(model_name, query, max_retries=5):
    current_task = "retrieval_query"
    for attempt in range(max_retries):
        try:
            return genai.embed_content(model=model_name, content=query, task_type=current_task)
        except Exception as err:
            msg = str(err)
            if "task_type" in msg.lower():
                current_task = None
                continue
            if "429" in msg or "ResourceExhausted" in msg:
                wait_s = retry_wait_seconds(msg)
                time.sleep(wait_s)
                continue
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("Failed to embed query after retries.")


@st.cache_resource(show_spinner=False)
def load_index(path):
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    texts = data["texts"]
    pages = data["pages"]
    sources = data["sources"]
    embed_model = str(data["embed_model"][0])
    return embeddings, texts, pages, sources, embed_model


def retrieve(query, embeddings, texts, pages, sources, embed_model, top_k=TOP_K):
    q = embed_query_with_retry(embed_model, query)
    qv = np.array(q["embedding"], dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    sims = embeddings @ qv
    idx = np.argsort(-sims)[:top_k]
    hits = []
    for i in idx:
        hits.append(
            {
                "score": float(sims[i]),
                "text": str(texts[i]),
                "page": int(pages[i]),
                "source": str(sources[i]),
            }
        )
    return hits


def answer_from_docs(query, hits, chat_model, min_sim=MIN_SIM):
    if not hits or hits[0]["score"] < min_sim:
        return "I could not find that in the provided documents."

    context_blocks = []
    for rank, h in enumerate(hits, start=1):
        context_blocks.append(
            f"[Chunk {rank} | source {h['source']} | page {h['page']} | score {h['score']:.3f}]\n{h['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are answering strictly from the supplied PDF excerpts.
Rules:
1) Use only facts present in CONTEXT.
2) If answer is not in CONTEXT, say exactly: I could not find that in the provided documents.
3) Do not use external knowledge.
4) End with citations like (source.pdf, page X).

QUESTION:
{query}

CONTEXT:
{context}
"""
    model = genai.GenerativeModel(chat_model)
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def get_api_key():
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key:
        return env_key

    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return ""


def main():
    st.set_page_config(page_title="SDI Assistant", page_icon=":books:", layout="wide")
    st.markdown(
        """
<style>
.stApp {
  background: radial-gradient(circle at 15% 20%, #f3f4f6 0%, #eef2ff 45%, #eaf2ff 100%);
  color: #0f172a;
}
p, li, label, span, div, h1, h2, h3, h4, h5, h6 {
  color: #0f172a;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
  color: #0f172a !important;
}
[data-testid="stTextInputRootElement"] label,
[data-testid="stTextInputRootElement"] input {
  color: #0f172a !important;
}
.hero {
  border: 1px solid #cbd5e1;
  border-radius: 16px;
  padding: 18px 20px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(3px);
}
.hero-title {
  font-size: 2rem;
  line-height: 1.2;
  font-weight: 700;
  color: #0b1220;
  margin: 0;
}
.hero-sub {
  color: #243247;
  margin-top: 8px;
  margin-bottom: 0;
}
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="hero">
  <p class="hero-title">SDI Document Assistant</p>
  <p class="hero-sub">Ask questions about the SDI Competition Guide. Answers are generated only from the indexed SDI documents.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if not os.path.exists(INDEX_FILE):
        st.error(
            f"Index file `{INDEX_FILE}` not found. Run `python build_index.py` locally and commit/upload `{INDEX_FILE}`."
        )
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error("Server configuration error: GEMINI_API_KEY is not set.")
        st.info("App owner: set GEMINI_API_KEY in Streamlit Secrets (Cloud) or environment variable (local).")
        st.stop()
    genai.configure(api_key=api_key)

    with st.spinner("Loading index..."):
        embeddings, texts, pages, sources, embed_model = load_index(INDEX_FILE)

    st.write(f"Indexed chunks: **{len(texts)}** | Docs: **{len(set(map(str, sources)))}**")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_hits" not in st.session_state:
        st.session_state.last_hits = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about SDI Competition Guide")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                hits = retrieve(query, embeddings, texts, pages, sources, embed_model, top_k=TOP_K)
                answer = answer_from_docs(query, hits, chat_model=CHAT_MODEL, min_sim=MIN_SIM)
                st.markdown(answer)
            st.session_state.last_hits = hits
        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("View source snippets used for the latest answer"):
        for i, h in enumerate(st.session_state.last_hits, start=1):
            st.markdown(f"**{i}. {h['source']} (page {h['page']})**")
            st.write(h["text"])


if __name__ == "__main__":
    main()
