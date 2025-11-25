"""
RAG Integration Module for Multi-Agent Orchestration System

This module provides RAG (Retrieval Augmented Generation) capabilities
to your agents, allowing them to query the knowledge base.
"""

from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Handles retrieval from ChromaDB vector database.
    
    Usage:
        retriever = RAGRetriever()
        context = retriever.retrieve("How do embeddings work?")
    """
    
    def __init__(self, db_path: str = None):
        """Initialize ChromaDB connection and embedding model"""
        # Default to project root / data / vector_db
        if db_path is None:
            project_root = Path(__file__).parent.parent
            db_path = project_root / "data" / "vector_db"
        
        self.db_path = Path(db_path)
        
        try:
            # Load ChromaDB with PersistentClient
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            
            # Get or create collection (matches ingestion script)
            self.collection = self.client.get_or_create_collection(
                name="ai_agent_knowledge_base",
                metadata={"description": "AI agents, embeddings, vector databases, LangGraph"}
            )
            
            # Load embedding model (same one used for ingestion)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        except Exception as e:
            print(f"❌ RAG Initialization Error: {e}")
            # We don't raise here to allow the agent to function without RAG if needed
            self.client = None
            self.collection = None
            self.embedding_model = None
    
    def retrieve(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Retrieve relevant chunks from knowledge base.
        
        Args:
            query: Natural language question
            n_results: Number of chunks to return (default: 3)
            
        Returns:
            List of dicts with 'content', 'source', 'chunk_id'
        """
        try:
            # Embed the query
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            # Format results
            chunks = []
            if results['documents']:
                for doc, meta, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    chunks.append({
                        'content': doc,
                        'source': meta.get('source', 'Unknown'),
                        'chunk_id': meta.get('chunk_id', 0),
                        'relevance': 1 - distance  # Note: Chroma default is L2, this is an approx relevance
                    })
            
            return chunks
            
        except Exception as e:
            print(f"❌ RAG Retrieval Error: {e}")
            # Return empty list on error so system doesn't crash
            return []
    
    def _expand_query(self, query: str) -> list[str]:
        """
        Expand user query into multiple search angles using LLM
        
        e.g., "What are types of agents?" becomes:
        - "agent taxonomies and classifications"
        - "orchestrator vs specialist agents"
        - "agent design patterns"
        """
        from utils.llm_factory import get_llm
        
        llm = get_llm(model_name="gemini-2.0-flash-exp", temperature=0.2)
        
        expansion_prompt = f"""Given this user question, generate 3-4 alternative phrasings or related queries that would help retrieve comprehensive information.

USER QUESTION: {query}

Generate queries that explore:
1. Different terminology (synonyms, technical terms)
2. Related concepts (parent/child topics)
3. Practical angles (use cases, examples)

OUTPUT FORMAT (just the queries, one per line):
<query 1>
<query 2>
<query 3>
"""
        
        try:
            response = llm.invoke(expansion_prompt)
            expanded = response.content.strip().split('\n')
            # Filter out empty lines and limit to 4
            expanded = [q.strip() for q in expanded if q.strip()][:4]
            logger.info(f"Query expanded into {len(expanded)} variants")
            return [query] + expanded  # Include original
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]
    
    def _rerank_results(self, query: str, results: list) -> list:
        """
        Re-rank retrieved results using LLM to score relevance
        
        This catches cases where vector similarity is misleading
        """
        if len(results) <= 3:
            return results  # Too few to re-rank
        
        from utils.llm_factory import get_llm
        llm = get_llm(model_name="gemini-2.0-flash-exp", temperature=0.0)
        
        # Format results for scoring
        result_text = "\n\n".join([
            f"CHUNK {i+1}:\n{r['text'][:300]}..."
            for i, r in enumerate(results)
        ])
        
        rerank_prompt = f"""Rate how relevant each chunk is to answering this question.

QUESTION: {query}

CHUNKS:
{result_text}

For each chunk, rate 0-10 (10 = directly answers, 0 = irrelevant).
OUTPUT FORMAT (just numbers, one per line):
<score for chunk 1>
<score for chunk 2>
...
"""
        
        try:
            response = llm.invoke(rerank_prompt)
            scores = [float(line.strip()) for line in response.content.strip().split('\n') if line.strip().replace('.','').isdigit()]
            
            # Combine original results with scores
            for i, score in enumerate(scores[:len(results)]):
                results[i]['rerank_score'] = score
            
            # Sort by rerank score
            results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.info(f"Re-ranked {len(results)} results")
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
        
        return results
    
    def retrieve_context_string(self, query: str, n_results: int = 5) -> str:
        """
        Advanced multi-stage retrieval with query expansion and re-ranking
        
        Pipeline:
        1. Expand query into multiple search angles
        2. Retrieve candidates from each angle
        3. Deduplicate and merge results
        4. Re-rank using LLM for true relevance
        5. Return top N
        """
        logger.info(f"Advanced retrieval for: {query}")
        
        # Stage 1: Query Expansion
        queries = self._expand_query(query)
        
        # Stage 2: Retrieve from multiple angles
        all_results = []
        seen_texts = set()
        
        for q in queries:
            # Embed the query
            query_embedding = self.embedding_model.encode([q]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results * 2  # Get more candidates
            )
            
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    # Deduplicate by text content
                    if doc not in seen_texts:
                        seen_texts.add(doc)
                        all_results.append({
                            'text': doc,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'distance': results['distances'][0][i] if results['distances'] else 1.0
                        })
        
        logger.info(f"Retrieved {len(all_results)} unique chunks from {len(queries)} query variants")
        
        # Stage 3: Re-rank for true relevance
        if len(all_results) > n_results:
            all_results = self._rerank_results(query, all_results)
        
        # Stage 4: Format top N results
        top_results = all_results[:n_results]
        
        context_parts = []
        for i, result in enumerate(top_results):
            source = result['metadata'].get('source', 'Unknown')
            rerank_score = result.get('rerank_score', 'N/A')
            
            context_parts.append(
                f"--- SOURCE {i+1}: {source} (Relevance: {rerank_score}) ---\n{result['text']}\n"
            )
        
        return "\n\n".join(context_parts)


# ============================================================================
# AGENT TOOL INTEGRATION
# ============================================================================

def create_rag_tool(retriever: RAGRetriever):
    """
    Create a tool definition that agents can use.
    
    This follows the standard tool format for LangChain, LangGraph, etc.
    """
    
    def search_knowledge_base(query: str) -> str:
        """
        Search the AI agent knowledge base for relevant information.
        
        Use this tool when you need information about:
        - Embeddings and vector search
        - AI agent architectures and patterns
        - RAG (Retrieval Augmented Generation) systems
        - Agent evaluation and monitoring
        - Fine-tuning and domain adaptation
        - MLOps and production deployment
        - Multi-agent systems
        - Function calling and tool use
        
        Args:
            query: Your search query (natural language question)
            
        Returns:
            Relevant information from the knowledge base with sources
        """
        return retriever.retrieve_context_string(query, n_results=3)
    
    # Tool definition (adapt to your framework)
    tool_definition = {
        "name": "search_knowledge_base",
        "description": search_knowledge_base.__doc__,
        "function": search_knowledge_base,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                }
            },
            "required": ["query"]
        }
    }
    
    return tool_definition


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize retriever
    retriever = RAGRetriever()
    
    # Example 1: Direct retrieval
    print("=" * 70)
    print("EXAMPLE 1: Direct Retrieval")
    print("=" * 70)
    
    chunks = retriever.retrieve("What are best practices for agent evaluation?")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{i}. {chunk['source']} (Relevance: {chunk['relevance']:.2f})")
        print(f"   {chunk['content'][:200]}...")
    
    # Example 2: Context string for LLM
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Context String for LLM")
    print("=" * 70)
    
    context = retriever.retrieve_context_string("How does agentic RAG work?")
    print(context[:500] + "...")
    
    # Example 3: As agent tool
    print("\n" + "=" * 70)
    print("EXAMPLE 3: As Agent Tool")
    print("=" * 70)
    
    tool = create_rag_tool(retriever)
    result = tool["function"]("What is the difference between PEFT and full fine-tuning?")
    print(result[:500] + "...")
