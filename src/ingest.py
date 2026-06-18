"""
ingest.py — Insurance Policy Document Ingestion Pipeline

Loads PDF documents, cleans text, chunks using RecursiveCharacterTextSplitter,
embeds using sentence-transformers, and stores in ChromaDB.

Run this script once to build the vector index:
    python src/ingest.py
"""

import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ── Configuration ──────────────────────────────────────────────
DATA_FOLDER = "data"
CHROMA_PATH = "data/chromadb"
COLLECTION_NAME = "insurance_policies"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

# ── Step 1: Load PDFs ──────────────────────────────────────────
def load_documents(data_folder):
    print("\n📂 Loading documents...")
    documents = {}
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_folder, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents[filename] = text
            print(f"  ✅ {filename}: {len(text)} characters")
    print(f"\n  Total documents loaded: {len(documents)}")
    return documents

# ── Step 2: Clean text ─────────────────────────────────────────
def clean_text(text):
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'([A-Z]{5,})\n', r'\1.\n\n', text)
    text = text.strip()
    return text

# ── Step 3: Chunk documents ────────────────────────────────────
def chunk_documents(documents):
    print("\n✂️  Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    all_chunks = []
    for filename, text in documents.items():
        cleaned = clean_text(text)
        chunks = text_splitter.split_text(cleaned)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "chunk_id": i
            })
        print(f"  ✅ {filename}: {len(chunks)} chunks")
    print(f"\n  Total chunks: {len(all_chunks)}")
    return all_chunks

# ── Step 4: Embed chunks ───────────────────────────────────────
def embed_chunks(all_chunks):
    print("\n🔢 Embedding chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )
    print(f"\n  Embeddings shape: {embeddings.shape}")
    return embeddings

# ── Step 5: Store in ChromaDB ──────────────────────────────────
def store_in_chromadb(all_chunks, embeddings):
    print("\n💾 Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete old collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Old collection deleted")
    except:
        print("  No existing collection found")

    # Create fresh collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data
    ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in all_chunks]
    texts_list = [chunk['text'] for chunk in all_chunks]
    embeddings_list = embeddings.tolist()
    metadatas = [
        {
            "source": chunk['source'],
            "chunk_id": chunk['chunk_id']
        }
        for chunk in all_chunks
    ]

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=texts_list,
        embeddings=embeddings_list,
        metadatas=metadatas
    )
    print(f"  ✅ Total chunks stored: {collection.count()}")
    return collection

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🛡️  Insurance Policy Ingestion Pipeline")
    print("="*50)

    documents = load_documents(DATA_FOLDER)
    all_chunks = chunk_documents(documents)
    embeddings = embed_chunks(all_chunks)
    collection = store_in_chromadb(all_chunks, embeddings)

    print("\n" + "="*50)
    print("✅ Ingestion complete!")
    print(f"   Documents: {len(documents)}")
    print(f"   Chunks: {len(all_chunks)}")
    print(f"   Embeddings: {embeddings.shape[0]} x {embeddings.shape[1]}")
    print(f"   ChromaDB: {collection.count()} items stored")
    print("="*50)