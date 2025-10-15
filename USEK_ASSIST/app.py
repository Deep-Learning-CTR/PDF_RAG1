# app.py — USEK Assistant (Public Info • RAG)
#   - Multilingual default embeddings (paraphrase-multilingual-MiniLM-L12-v2)
#   - Persistent index (Chroma/FAISS) auto-loads at startup
#   - Rebuild only when you change embedding/chunk params or click the button
# Requirements:
#   pip install -U streamlit sentence-transformers chromadb faiss-cpu python-dotenv openai google-generativeai cerebras-cloud-sdk

import os, pickle, time, json
from pathlib import Path
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Vector stores
import chromadb
from chromadb.config import Settings
import faiss

# LLM providers
from openai import OpenAI as OpenAIClient
from openai import OpenAIError, RateLimitError
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, NotFound
from cerebras.cloud.sdk import Cerebras

# ---------- Setup ----------
load_dotenv(dotenv_path=Path(__file__).with_name('.env'))
st.set_page_config(page_title="USEK Assistant", layout="wide")
st.title("USEK Assistant (Public Info • RAG)")

INDEX_DIR = Path("indexes")
DATA_DIR  = Path("data")
MANIFEST  = INDEX_DIR / "manifest.json"

# Default embedding (multilingual)
DEFAULT_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---------- Sidebar Controls ----------
st.sidebar.header("Settings")

store_type = st.sidebar.selectbox("Vector store", ["ChromaDB", "FAISS"])
provider   = st.sidebar.selectbox("LLM provider", ["Gemini", "Cerebras", "OpenAI"])

# Optional model overrides (leave blank to auto-select)
gemini_model_input   = st.sidebar.text_input("Gemini model (optional)", value="")
cerebras_model_input = st.sidebar.text_input("Cerebras model (optional)", value="")
openai_model_input   = st.sidebar.text_input("OpenAI model (optional)", value="")

top_k       = st.sidebar.slider("Top-K (retrieval)", 1, 15, 5)
temperature = st.sidebar.slider("Temperature (LLM)", 0.0, 1.0, 0.2)

# Embeddings
embed_model_name = st.sidebar.text_input("Embedding model", value=DEFAULT_EMBED)

# Persist/Auto-rebuild behavior
auto_rebuild_if_missing = st.sidebar.checkbox("Auto-build index if missing (recommended)", value=True)

st.sidebar.subheader("Rebuild Index")
chunk_size    = st.sidebar.number_input("Chunk size", 300, 2000, 800, 50)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 800, 100, 10)
rebuild       = st.sidebar.button("Rebuild from data/*.json")
st.sidebar.markdown("**Note:** Rebuild reads files in `data/`. Update them with `python crawler.py`.")

# ---------- Helpers ----------
MAX_TURNS = 3  # keep last 3 Q&A turns in prompt

def require_env(varname: str):
    val = os.getenv(varname)
    if not val or val.strip() == "":
        st.error(f"Missing {varname}. Set it in Windows Environment Variables (Win+R → sysdm.cpl ,3) or .env.")
        st.stop()
    return val

def clamp_history():
    msgs = st.session_state.history
    if len(msgs) > MAX_TURNS * 2:
        st.session_state.history = msgs[-MAX_TURNS*2:]

def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        if chunk.strip():
            chunks.append(chunk)
        i = end - overlap if end - overlap > i else end
    return chunks

@st.cache_resource(show_spinner=False)
def load_encoder(name: str):
    return SentenceTransformer(name)

def encode_one(text, encoder):
    return encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

def encode_many(texts, encoder):
    return encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def get_chroma_client():
    return chromadb.PersistentClient(
        path=str(INDEX_DIR / "chroma"),
        settings=Settings(anonymized_telemetry=False)
    )

def read_manifest():
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def write_manifest(d):
    INDEX_DIR.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def manifest_needs_rebuild(current):
    m = read_manifest()
    if not m:
        return True
    keys = ["store_type", "embedding_model", "chunk_size", "chunk_overlap"]
    for k in keys:
        if str(m.get(k)) != str(current.get(k)):
            return True
    # If data folder is empty, rebuild anyway
    if not list(DATA_DIR.glob("*.json")):
        return True
    return False

# ---------- Rebuild index ----------
def rebuild_index():
    import json
    status = st.empty()
    status.write("Step 1/4: Reading data files…")

    files = list(DATA_DIR.glob("*.json"))
    if not files:
        st.warning("No files in data/. Run: python crawler.py")
        return False

    # load docs
    docs = []
    for f in files:
        obj = json.loads(f.read_text(encoding="utf-8"))
        md = obj.get("markdown", "")
        if md.strip():
            docs.append({"url": obj.get("url",""), "title": obj.get("title",""), "text": md})

    if not docs:
        st.warning("Found JSON files but no Markdown content inside them.")
        return False

    status.write("Step 2/4: Chunking…")
    chunks, metas = [], []
    for d in docs:
        parts = chunk_text(d["text"], chunk_size, chunk_overlap)
        for p in parts:
            chunks.append(p)
            metas.append({"source": d["url"], "title": d["title"]})

    status.write("Step 3/4: Embedding… (first run may take longer)")
    enc = load_encoder(embed_model_name)
    embs = encode_many(chunks, enc)

    INDEX_DIR.mkdir(exist_ok=True)

    status.write("Step 4/4: Writing index to disk…")
    if store_type == "ChromaDB":
        client = get_chroma_client()
        try:
            client.delete_collection("usek")
        except Exception:
            pass
        col = client.create_collection("usek")
        import uuid
        ids = [str(uuid.uuid4()) for _ in chunks]
        col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embs.tolist())
        st.success(f"Rebuilt Chroma index with {len(chunks)} chunks.")
    else:
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine ≈ inner product with normalized vectors
        index.add(embs.astype(np.float32))
        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "faiss_meta.pkl", "wb") as f:
            pickle.dump({"chunks": chunks, "metadatas": metas}, f)
        st.success(f"Rebuilt FAISS index with {len(chunks)} chunks.")

    # Save manifest so future runs auto-load without rebuilding
    write_manifest({
        "store_type": store_type,
        "embedding_model": embed_model_name,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "built_at": time.time(),
    })

    status.empty()
    return True

# Spinner around manual rebuild
if rebuild:
    with st.spinner("Rebuilding index… this can take a few minutes on first run"):
        try:
            start = time.time()
            ok = rebuild_index()
            if ok:
                st.info(f"Rebuild finished in {time.time() - start:.1f}s")
        except ValueError:
            st.error("Rebuild failed due to an old Chroma DB. Close app, delete `indexes\\chroma`, reopen, and Rebuild.")
            st.stop()

# ---------- Load/ensure store ----------
def faiss_files_exist():
    return (INDEX_DIR / "faiss.index").exists() and (INDEX_DIR / "faiss_meta.pkl").exists()

def chroma_collection_exists():
    try:
        client = get_chroma_client()
        client.get_collection("usek")
        return True
    except Exception:
        return False

def ensure_index_ready():
    """Auto-build if missing or manifest mismatch, else do nothing."""
    current = {
        "store_type": store_type,
        "embedding_model": embed_model_name,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
    }

    missing = False
    if store_type == "ChromaDB":
        missing = not chroma_collection_exists()
    else:
        missing = not faiss_files_exist()

    needs = missing or manifest_needs_rebuild(current)

    if needs and auto_rebuild_if_missing:
        with st.spinner("Index not ready → building it now…"):
            ok = rebuild_index()
            if not ok:
                st.stop()
    elif needs and not auto_rebuild_if_missing:
        st.warning("Index not ready. Click 'Rebuild' to build it with current settings.")
        st.stop()

ensure_index_ready()

def load_store():
    enc = load_encoder(embed_model_name)

    if store_type == "ChromaDB":
        client = get_chroma_client()
        try:
            col = client.get_collection("usek")
        except Exception:
            st.error("Chroma collection not found. Click 'Rebuild' or enable auto-build.")
            return None, enc
        return ("chroma", col), enc

    else:
        faiss_path = INDEX_DIR / "faiss.index"
        meta_path  = INDEX_DIR / "faiss_meta.pkl"
        if not faiss_path.exists() or not meta_path.exists():
            st.error("FAISS files not found. Click 'Rebuild' or enable auto-build.")
            return None, enc
        index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return ("faiss", (index, meta)), enc

store, encoder = load_store()

def retrieve(query: str, k: int):
    if store is None:
        return []
    q_emb = encode_one(query, encoder)

    if store[0] == "chroma":
        col = store[1]
        r = col.query(query_embeddings=[q_emb.tolist()], n_results=k, include=["documents","metadatas","distances"])
        outs = []
        if r["documents"]:
            for i in range(len(r["documents"][0])):
                outs.append({
                    "text": r["documents"][0][i],
                    "source": r["metadatas"][0][i].get("source",""),
                    "title": r["metadatas"][0][i].get("title",""),
                    "score": r["distances"][0][i],
                })
        return outs
    else:
        index, meta = store[1]
        D, I = index.search(np.array([q_emb]).astype(np.float32), k)
        outs = []
        for rank, idx in enumerate(I[0]):
            if idx < 0: continue
            outs.append({
                "text": meta["chunks"][idx],
                "source": meta["metadatas"][idx].get("source",""),
                "title": meta["metadatas"][idx].get("title",""),
                "score": float(D[0][rank]),
            })
        return outs

# ---------- Prompt building (includes last 3 Q&A turns) ----------
def build_prompt(question: str, ctx):
    # 1) recent dialogue (last 3 Q&A)
    dialogue_lines = []
    for m in st.session_state.history[-MAX_TURNS*2:]:
        if m["role"] == "user":
            dialogue_lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            dialogue_lines.append(f"Assistant: {m['content']}")
    dialogue_block = "\n".join(dialogue_lines) if dialogue_lines else "No prior turns."

    # 2) retrieved snippets
    bullets = []
    for i, c in enumerate(ctx, 1):
        bullets.append(f"[{i}] Source: {c['source']}\n{c['text']}")
    ctxt = "\n\n---\n\n".join(bullets) if bullets else "(no snippets found)"

    system = (
        "You are a helpful assistant that must answer ONLY from the provided public context. "
        "If the answer is not in context or seems to require private/portal access, say you don’t have access. "
        "Be concise, fluent, and add inline citations like [1], [2] matching the snippets list."
    )
    user = (
        f"Recent dialogue:\n{dialogue_block}\n\n"
        f"Question: {question}\n\n"
        f"Context snippets:\n{ctxt}\n\n"
        "Answer in English with citations."
    )
    return system, user

# ---------- Auto model selection ----------
def auto_model_for_gemini(requested: str | None) -> str:
    api_key = require_env("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        usable = [m.name.split("/")[-1] for m in models if "generateContent" in m.supported_generation_methods]
    except Exception:
        usable = []  # if listing fails, we’ll just use defaults
    candidates = []
    if requested and requested.strip():
        candidates.append(requested.strip())
    candidates += ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]
    for c in candidates:
        if not usable or c in usable:
            return c
    return candidates[0]

def auto_model_for_cerebras(requested: str | None) -> str:
    candidates = []
    if requested and requested.strip():
        candidates.append(requested.strip())
    candidates += ["llama3.1-8b", "llama3.1-70b", "mixtral-8x7b"]
    return candidates[0]

def auto_model_for_openai(requested: str | None) -> str:
    if requested and requested.strip():
        return requested.strip()
    return "gpt-4o-mini"

# ---------- LLM call ----------
def llm_answer(provider, system_prompt, user_prompt, temperature):
    if provider == "Gemini":
        api_key = require_env("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        model_name = auto_model_for_gemini(gemini_model_input)
        try:
            model_obj = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
            out = model_obj.generate_content(user_prompt, generation_config={"temperature": float(temperature)})
            return out.text or "No response."
        except NotFound:
            try:
                model_obj = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", system_instruction=system_prompt)
                out = model_obj.generate_content(user_prompt, generation_config={"temperature": float(temperature)})
                return out.text or "No response."
            except GoogleAPIError as ge:
                st.error(f"Gemini error: {getattr(ge, 'message', ge)}")
                return "Gemini call failed."
        except GoogleAPIError as ge:
            st.error(f"Gemini error: {getattr(ge, 'message', ge)}")
            return "Gemini call failed."

    if provider == "Cerebras":
        api_key = require_env("CEREBRAS_API_KEY")
        client = Cerebras(api_key=api_key)
        model_name = auto_model_for_cerebras(cerebras_model_input)
        try:
            chat = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=float(temperature),
            )
            if chat.choices:
                return chat.choices[0].message.content
            return "No response."
        except Exception as e:
            st.error(f"Cerebras error: {e}")
            return "Cerebras call failed."

    # OpenAI
    api_key = require_env("OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key)
    model_name = auto_model_for_openai(openai_model_input)
    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=float(temperature),
        )
        return resp.output_text
    except RateLimitError:
        st.error("OpenAI: insufficient quota/rate limited. Add billing/credit or switch provider in the sidebar.")
        return "Sorry, the OpenAI account has no quota."
    except OpenAIError as e:
        st.error(f"OpenAI error: {e}")
        return "OpenAI call failed."

# ---------- Chat UI ----------
if "history" not in st.session_state:
    st.session_state.history = []

# show previous messages
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask about public info on www.usek.edu.lb ...")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            ctx = retrieve(q, top_k)
            sys_p, user_p = build_prompt(q, ctx)

        with st.spinner(f"Asking {provider}…"):
            ans = llm_answer(provider, sys_p, user_p, temperature)
            st.markdown(ans)

        # Retrieved snippets viewer
        if ctx:
            options = [f"[{i+1}] {c['title'] or 'Page'} — {c['source']}" for i, c in enumerate(ctx)]
            chosen = st.selectbox("Retrieved snippets (select to view):", options, index=0)
            idx = options.index(chosen)
            with st.expander("Snippet content", expanded=True):
                st.write(f"**Source:** {ctx[idx]['source']}")
                st.text_area("Snippet", ctx[idx]["text"], height=220)
        else:
            st.info("No snippets retrieved. Try a simpler question.")

    st.session_state.history.append({"role": "assistant", "content": ans})
    clamp_history()
