"""
Re-ingest RAG database with improved chunking strategy.

This script re-processes the knowledge base with:
1. Larger chunk sizes (1000 chars vs 500)
2. Header-aware chunking (preserves Markdown structure)
3. Better metadata (includes section titles)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_retriever import RAGRetriever
import chromadb
from sentence_transformers import SentenceTransformer
import re

def smart_chunk_markdown(content: str, chunk_size=1000, overlap=100):
    """
    Chunk Markdown content intelligently, preserving headers.
    """
    chunks = []
    current_chunk = ""
    current_header = ""
    
    lines = content.split('\n')
    
    for line in lines:
        # Detect markdown headers
        if line.startswith('#'):
            current_header = line.strip()
        
        if len(current_chunk) + len(line) > chunk_size:
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'header': current_header
                })
                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap:] + '\n' + line
            else:
                current_chunk = line
        else:
            current_chunk += '\n' + line
    
    if current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'header': current_header
        })
    
    return chunks

def reingest():
    print("🔄 Re-ingesting knowledge base with improved chunking...")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "final"
    db_path = project_root / "data" / "vector_db"
    
    # Delete old database
    import shutil
    if db_path.exists():
        print(f"🗑️  Deleting old database at {db_path}")
        shutil.rmtree(db_path)
    
    # Create new client
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name="ai_agent_knowledge_base",
        metadata={"description": "AI agents knowledge base - improved chunking"}
    )
    
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process all files
    all_chunks = []
    all_metadatas = []
    
    for file_path in data_dir.glob("*.md"):
        print(f"📄 Processing: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Smart chunking
        chunks = smart_chunk_markdown(content, chunk_size=1000, overlap=100)
        
        for i, chunk_data in enumerate(chunks):
            all_chunks.append(chunk_data['text'])
            all_metadatas.append({
                'source': file_path.name,
                'chunk_id': i,
                'section_header': chunk_data['header']
            })
        
        print(f"   → Generated {len(chunks)} chunks")
    
    print(f"\n✂️  Total chunks: {len(all_chunks)}")
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True).tolist()
    
    # Add to database in batches
    print("💾 Adding to vector database...")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        
        collection.add(
            documents=all_chunks[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=all_metadatas[i:batch_end],
            ids=[f"chunk_{j}" for j in range(i, batch_end)]
        )
        
        print(f"   Batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
    
    print(f"\n✅ Re-ingestion complete!")
    print(f"📊 Total documents: {len(set(m['source'] for m in all_metadatas))}")
    print(f"📊 Total chunks: {collection.count()}")
    
    # Test retrieval
    print("\n🧪 Testing retrieval...")
    results = collection.query(
        query_embeddings=[embedding_model.encode(["types of agents taxonomy"]).tolist()[0]],
        n_results=3
    )
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {meta['source']}")
        print(f"Header: {meta.get('section_header', 'N/A')}")
        print(f"Content: {doc[:150]}...")

if __name__ == "__main__":
    reingest()
