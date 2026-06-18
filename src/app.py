import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
import httpx

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Policy Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base theme */
    .stApp {
        background-color: #0A1628;
        color: #E8EDF5;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0D1F3C;
        border-right: 1px solid #1E3A5F;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0D1F3C 0%, #1A3A6B 100%);
        border: 1px solid #C9A84C;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 14px;
        color: #8FA3BF;
        margin: 4px 0 0 0;
    }
    
    .gold-accent {
        color: #C9A84C;
    }

    /* Search box */
    .stTextArea textarea {
        background-color: #0D1F3C !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px !important;
        color: #E8EDF5 !important;
        font-size: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #C9A84C !important;
        box-shadow: 0 0 0 1px #C9A84C !important;
    }

    /* Answer card */
    .answer-card {
        background-color: #0D1F3C;
        border: 1px solid #1E3A5F;
        border-left: 4px solid #C9A84C;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        line-height: 1.7;
        font-size: 15px;
        color: #E8EDF5;
    }
    
    /* Source citation cards */
    .source-card {
        background-color: #112240;
        border: 1px solid #1E3A5F;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #8FA3BF;
    }
    
    .source-label {
        color: #C9A84C;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #C9A84C, #A8873D);
        color: #0A1628;
        border: none;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        padding: 10px 24px;
        width: 100%;
        transition: opacity 0.2s;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        color: #0A1628;
    }

    /* Document pills in sidebar */
    .doc-pill {
        background-color: #112240;
        border: 1px solid #1E3A5F;
        border-radius: 20px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 12px;
        color: #8FA3BF;
        display: block;
    }
    
    /* Stats bar */
    .stat-box {
        background-color: #0D1F3C;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    
    .stat-number {
        font-size: 24px;
        font-weight: 700;
        color: #C9A84C;
    }
    
    .stat-label {
        font-size: 11px;
        color: #8FA3BF;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* History items */
    .history-item {
        background-color: #112240;
        border: 1px solid #1E3A5F;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #8FA3BF;
        cursor: pointer;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider */
    hr {
        border-color: #1E3A5F;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models and DB ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    load_dotenv()
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    chroma_path = "data/chromadb"
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(
        name="insurance_policies",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Auto-run ingestion if collection is empty
# Auto-run ingestion if collection is empty
if collection.count() == 0:
    st.info("⏳ Building document index for first time... please wait (this takes 2-3 minutes)")
    
    import re
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Load PDFs
    data_folder = "data"
    documents = {}
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_folder, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents[filename] = text
    
    # Clean text
    def clean_text(text):
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'([A-Z]{5,})\n', r'\1.\n\n', text)
        return text.strip()
    
    # Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    all_chunks = []
    for filename, text in documents.items():
        chunks = text_splitter.split_text(clean_text(text))
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "chunk_id": i
            })
    
    # Embed
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = embedding_model.encode(texts, batch_size=32)
    
    # Store
    ids = [f"{c['source']}_{c['chunk_id']}" for c in all_chunks]
    texts_list = [c['text'] for c in all_chunks]
    embeddings_list = embeddings.tolist()
    metadatas = [{"source": c['source'], "chunk_id": c['chunk_id']} for c in all_chunks]
    
    collection.add(
        ids=ids,
        documents=texts_list,
        embeddings=embeddings_list,
        metadatas=metadatas
    )
    
    st.success(f"✅ Index built! {collection.count()} chunks indexed.")
    
    ssl_verify = os.getenv("SSL_VERIFY", "true").lower() != "false"
    groq_client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
        http_client=httpx.Client(verify=ssl_verify)
    )
    return embedding_model, collection, groq_client

embedding_model, collection, groq_client = load_resources()

# ── Core RAG function ──────────────────────────────────────────
def ask_question(question, n_results=3):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    retrieved_chunks = [
        {
            "source": results['metadatas'][0][i]['source'],
            "chunk_id": results['metadatas'][0][i]['chunk_id'],
            "text": results['documents'][0][i]
        }
        for i in range(n_results)
    ]
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n\n"

    prompt = f"""You are an expert insurance policy assistant for a professional insurance company.
Answer the question below using ONLY the context provided from the policy documents.
If the answer is not in the context, say "I could not find this in the provided policy documents."
Always cite which source document your answer comes from.
Be concise, professional, and precise.

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return answer, retrieved_chunks

# ── Session state ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding: 8px 0 16px 0;'>
            <span style='font-size:22px;'>🛡️</span>
            <span style='font-size:16px; font-weight:700; 
            color:#FFFFFF; margin-left:8px;'>Policy Assistant</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class='stat-box'>
                <div class='stat-number'>6</div>
                <div class='stat-label'>Documents</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='stat-box'>
                <div class='stat-number'>2375</div>
                <div class='stat-label'>Chunks</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Document list
    st.markdown("""
        <div style='font-size:11px; color:#8FA3BF; 
        text-transform:uppercase; letter-spacing:0.8px; 
        margin-bottom:8px;'>Loaded Documents</div>
    """, unsafe_allow_html=True)
    
    docs = [
        "commercial_combined_insurance_policy",
        "freight_solutions_policy",
        "legacy_policy",
        "office_policy",
        "property_owners_policy",
        "property_policy"
    ]
    for doc in docs:
        st.markdown(f"""
            <div class='doc-pill'>📄 {doc.replace('_', ' ')}</div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Query history
    st.markdown("""
        <div style='font-size:11px; color:#8FA3BF; 
        text-transform:uppercase; letter-spacing:0.8px; 
        margin-bottom:8px;'>Recent Queries</div>
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        for item in st.session_state.history[-5:][::-1]:
            st.markdown(f"""
                <div class='history-item'>
                    💬 {item['question'][:45]}...
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='font-size:12px; color:#4A6080;
            font-style:italic;'>No queries yet</div>
        """, unsafe_allow_html=True)

# ── Main content ───────────────────────────────────────────────
st.markdown("""
    <div class='main-header'>
        <p class='main-title'>
            Insurance Policy <span class='gold-accent'>Intelligence</span>
        </p>
        <p class='main-subtitle'>
            Ask questions across 6 policy documents — 
            answers grounded in source documents with citations
        </p>
    </div>
""", unsafe_allow_html=True)

# Suggested questions
st.markdown("""
    <div style='font-size:11px; color:#8FA3BF; 
    text-transform:uppercase; letter-spacing:0.8px; 
    margin-bottom:10px;'>Suggested Questions</div>
""", unsafe_allow_html=True)

suggestions = [
    "What is excluded from storm or flood damage?",
    "What are the conditions for making a claim?",
    "Is theft covered under the office policy?",
    "What is the liability coverage limit?"
]

cols = st.columns(4)
for i, suggestion in enumerate(suggestions):
    with cols[i]:
        if st.button(suggestion, key=f"suggest_{i}"):
            st.session_state.selected_question = suggestion

# Question input
question = st.text_area(
    "Ask a question about your insurance policies",
    value=st.session_state.get("selected_question", ""),
    placeholder="e.g. What is covered under property damage?",
    height=80,
    label_visibility="collapsed"
)

col_btn, col_clear = st.columns([4, 1])
with col_btn:
    search_clicked = st.button("🔍 Search Policy Documents")
with col_clear:
    if st.button("Clear"):
        st.session_state.selected_question = ""
        st.rerun()

# ── Answer display ─────────────────────────────────────────────
if search_clicked and question.strip():
    with st.spinner("Searching policy documents..."):
        answer, chunks = ask_question(question)
    
    # Save to history
    st.session_state.history.append({
        "question": question,
        "answer": answer,
        "sources": [c['source'] for c in chunks]
    })
    
    # Display answer
    st.markdown(f"""
        <div class='answer-card'>
            {answer}
        </div>
    """, unsafe_allow_html=True)
    
    # Display sources
    st.markdown("""
        <div style='font-size:11px; color:#8FA3BF; 
        text-transform:uppercase; letter-spacing:0.8px; 
        margin: 16px 0 8px 0;'>Source Documents Retrieved</div>
    """, unsafe_allow_html=True)
    
    for i, chunk in enumerate(chunks):
        st.markdown(f"""
            <div class='source-card'>
                <div class='source-label'>Source {i+1}</div>
                <div style='color:#C9A84C; margin: 4px 0;'>
                    📄 {chunk['source'].replace('_', ' ').replace('.pdf', '')}
                </div>
                <div style='margin-top: 6px; 
                font-size:12px;'>{chunk['text'][:200]}...</div>
            </div>
        """, unsafe_allow_html=True)

elif search_clicked and not question.strip():
    st.warning("Please enter a question first.")

# Previous answers
if len(st.session_state.history) > 1:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:11px; color:#8FA3BF; 
        text-transform:uppercase; letter-spacing:0.8px; 
        margin-bottom:12px;'>Previous Answers</div>
    """, unsafe_allow_html=True)
    
    for item in st.session_state.history[:-1][::-1]:
        with st.expander(f"💬 {item['question']}"):
            st.markdown(f"""
                <div class='answer-card'>{item['answer']}</div>
            """, unsafe_allow_html=True)