# 🛡️ Insurance Policy Intelligence Assistant

🚀 Live Demo: https://insurance-policy-assistant.streamlit.app/

An AI-powered Q&A system that answers questions from insurance policy 
documents using Retrieval-Augmented Generation (RAG). Built as a 
portfolio project targeting AI engineering roles in the insurance industry.

---

## 🎯 What This Does

Instead of manually searching through hundreds of pages of policy 
documents, this tool lets you ask plain-English questions and get 
back cited answers grounded in the actual policy text.

**Example questions:**
- "What is excluded from storm or flood damage?"
- "Is theft covered under the office policy?"
- "What are the conditions for making a claim?"
- "What is the liability coverage limit?"

Each answer includes citations showing exactly which document and 
section the information came from.

---

## 🏗️ Architecture

PDF Documents
↓
Text Extraction (pypdf)
↓
Chunking (500 char chunks, 50 char overlap)
↓
Embeddings (sentence-transformers: all-MiniLM-L6-v2, 384 dimensions)
↓
Vector Storage (ChromaDB with cosine similarity)
↓
Semantic Retrieval (top 3 chunks per query)
↓
LLM Generation (Llama 3.3 70B via Groq API)
↓
Cited Answer + Source Cards (Streamlit UI)

## 🗂️ Project Structure

insurance-policy-assistant/
data/                   ← Add your PDF documents here (not in Git)
notebooks/
    01_explore_pdfs.ipynb ← Step by step: load, chunk, embed, store, query
src/
    app.py                ← Streamlit UI
evaluation/             ← RAGAS evaluation results (coming soon)
.env                    ← Your API keys (never committed)
requirements.txt        ← All dependencies

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF Extraction | pypdf | Reliable text extraction from complex PDFs |
| Chunking | Custom (500 char + 50 overlap) | Balances context and retrieval precision |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, local, 384-dim semantic vectors |
| Vector Database | ChromaDB (persistent) | Fast cosine similarity search, no server needed |
| LLM | Llama 3.3 70B via Groq | Free, fast, strong instruction following |
| UI | Streamlit | Rapid prototyping with professional styling |
| Evaluation | RAGAS | Faithfulness, relevance, context recall metrics |

---

## 🚀 How to Run

**1. Clone the repo:**
```bash
git clone https://github.com/YOUR-USERNAME/insurance-policy-assistant.git
cd insurance-policy-assistant
```

**2. Create and activate virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate.bat
# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Add your documents:**
- Create a `data/` folder
- Add insurance policy PDFs to `data/`

**5. Add your Groq API key:**
- Create a `.env` file
- Add: `GROQ_API_KEY=your_key_here`
- Get a free key at: https://console.groq.com

**6. Run the notebook to index your documents:**
- Open `notebooks/01_explore_pdfs.ipynb`
- Run all cells to chunk, embed and store documents in ChromaDB

**7. Launch the app:**
```bash
streamlit run src/app.py
```

---

## 📊 Documents Indexed

| Document | Size |
|---|---|
| Commercial Combined Insurance Policy | 332,617 chars |
| Freight Solutions Policy | 60,192 chars |
| Legacy Policy | 215,696 chars |
| Office Policy | 248,582 chars |
| Property Owners Policy | 108,299 chars |
| Property Policy | 101,860 chars |

**Total:** 6 documents · 2,375 chunks · 1,067,146 characters

---

## 🔍 How RAG Works Here

1. **Chunking** — Each PDF is split into 500-character chunks with 
50-character overlap to avoid cutting sentences at boundaries
2. **Embedding** — Each chunk is converted to a 384-dimensional vector 
capturing its semantic meaning using a local sentence transformer model
3. **Storage** — All vectors stored in ChromaDB with cosine similarity 
index for fast nearest-neighbor search
4. **Retrieval** — User question is embedded and top 3 most semantically 
similar chunks are retrieved
5. **Generation** — Retrieved chunks + question sent to Llama 3.3 70B 
with grounding instructions to prevent hallucination
6. **Citation** — Answer displayed with source document references

---

## 📊 Evaluation Results

Evaluated using an LLM-as-a-judge approach (RAGAS had unresolvable 
dependency conflicts on Windows — see Project FAQ for details) across 
15 test questions:

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.71 |
| Overall | 0.81 |

4 of 15 questions returned "could not find" responses — all retrieval 
failures (system correctly avoided hallucinating), not generation 
failures. See `evaluation/ragas_results.csv` for full breakdown.

## ⚠️ What I Learned / What Failed

**What worked well:**
- Semantic search correctly retrieved relevant chunks even when exact 
keywords weren't present (e.g. "burst pipe" matched "escape of water" coverage)
- Grounding instructions effectively prevented hallucination — the model 
said "I could not find this" rather than inventing answers
- ChromaDB's persistent storage means embeddings are computed once and 
reused across sessions

**What failed / limitations:**
- Fixed character chunking caused retrieval gaps — answers split across 
chunk boundaries were sometimes missed (e.g. "What is covered under 
property damage?" returned only exclusions)
- Chunk size of 500 characters is sometimes too small for complex 
multi-part policy clauses

**Planned improvements:**
- Switch to sentence-aware or section-aware chunking (LangChain's 
RecursiveCharacterTextSplitter)
- Add RAGAS evaluation scores with a documented test set
- Add re-ranking layer to improve retrieval precision
- Support document upload via UI

---

## 🔮 Relevance to Insurance Industry

This project directly addresses four capability areas relevant to 
modern insurance operations:

- **Intelligent Document Processing** — automated extraction and 
retrieval from dense policy documents
- **AI Tool Building for Underwriting** — assistants that help 
underwriters answer policy questions instantly
- **Information Retrieval** — semantic search across heterogeneous 
document types
- **LLM-Assisted Workflows** — grounded generation with source 
attribution to maintain compliance and auditability

---

## 📈 Next Steps (Phase 2 Enhancements)

- [ ] RAGAS evaluation with documented test questions and scores
- [ ] Sentence-aware chunking to fix boundary issues
- [ ] Re-ranking with cross-encoder for better retrieval precision
- [ ] Multi-turn conversation support
- [ ] Document upload feature in UI

---

*Data source: Policy documents from 
[Intact Insurance NI](https://intactinsuranceni.com/existing-customers/existing-non-motor-customers) 
— publicly available policy wordings.*