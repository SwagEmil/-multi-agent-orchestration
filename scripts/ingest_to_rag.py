#!/usr/bin/env python3
"""
Final step: Ingest cleaned documents into ChromaDB for RAG.

Usage:
    python scripts/ingest_to_rag.py --docs-dir data/final/
"""

import argparse
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re

class RAGIngester:
    def __init__(self, db_path="data/vector_db"):
        """Initialize ChromaDB and embedding model"""
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        print("🔧 Initializing RAG system...")
        print(f"   Database: {self.db_path}")
        
        # Initialize ChromaDB with PersistentClient
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Load embedding model (free, runs locally)
        print("   Loading embedding model (sentence-transformers)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="ai_agent_knowledge_base",
            metadata={"description": "AI agents, embeddings, vector databases, LangGraph"}
        )
        
        print("✅ RAG system initialized\n")
    
    def chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split document into overlapping chunks"""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', content)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def ingest_document(self, file_path: Path):
        """Ingest a single document"""
        print(f"📄 Ingesting: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from first few lines
        lines = content.split('\n', 10)
        metadata = {
            'source': file_path.name,
            'path': str(file_path)
        }
        
        # Try to extract title
        for line in lines:
            if line.startswith('Title:'):
                metadata['title'] = line.replace('Title:', '').strip()
            elif line.startswith('**Source:**'):
                metadata['url'] = line.replace('**Source:**', '').strip()
            elif line.startswith('**Topic:**'):
                metadata['topic'] = line.replace('**Topic:**', '').strip()
        
        # Chunk the document
        chunks = self.chunk_document(content)
        
        print(f"   Generated {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"   Generating embeddings...")
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Create unique IDs
        doc_id = file_path.stem
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Prepare metadata for each chunk
        metadatas = [
            {**metadata, 'chunk_id': i, 'total_chunks': len(chunks)}
            for i in range(len(chunks))
        ]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"   ✅ Ingested {len(chunks)} chunks\n")
        
        return len(chunks)
    
    def ingest_directory(self, directory: str):
        """Ingest all markdown files from directory"""
        dir_path = Path(directory)
        
        print(f"\n{'='*70}")
        print(f"RAG INGESTION - AI AGENT KNOWLEDGE BASE")
        print(f"{'='*70}\n")
        
        files = list(dir_path.glob('*.md'))
        
        if not files:
            print(f"⚠️  No .md files found in {directory}")
            return
        
        print(f"📊 Found {len(files)} documents to ingest\n")
        
        total_chunks = 0
        for file_path in files:
            chunks = self.ingest_document(file_path)
            total_chunks += chunks
        
        # Print summary
        print(f"{'='*70}")
        print(f"INGESTION COMPLETE")
        print(f"{'='*70}\n")
        print(f"✅ Total documents: {len(files)}")
        print(f"✅ Total chunks: {total_chunks}")
        print(f"✅ Database location: {self.db_path}")
        print(f"\n💡 Your knowledge base is ready for RAG queries!\n")
    
    def test_retrieval(self, query: str, n_results: int = 3):
        """Test retrieval with a sample query"""
        print(f"\n🔍 Testing retrieval: '{query}'\n")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        print(f"Top {n_results} results:\n")
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            relevance = 1 - dist  # Convert distance to similarity
            print(f"{i}. [Relevance: {relevance:.2f}] {meta.get('source', 'Unknown')}")
            print(f"   {doc[:200]}...\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-dir', required=True, help='Directory containing cleaned documents')
    parser.add_argument('--test', action='store_true', help='Run test query after ingestion')
    args = parser.parse_args()
    
    # Ingest documents
    ingester = RAGIngester()
    ingester.ingest_directory(args.docs_dir)
    
    # Optional: Test retrieval
    if args.test:
        ingester.test_retrieval("How do embeddings work for agent systems?")

if __name__ == "__main__":
    main()
