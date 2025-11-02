"""
RAG Tool for Google ADK Agents
This tool allows agents to query the compliance document RAG system.
"""
import logging
from typing import List, Dict, Optional
from rag_service import get_rag_service

logger = logging.getLogger(__name__)


def compliance_rag_search(query: str, n_results: int = 5) -> str:
    """
    Search compliance documents using RAG (Retrieval-Augmented Generation).
    
    This tool searches through your uploaded compliance documents to find
    relevant information based on the query. Use this instead of generic
    web search when you need specific information from compliance documents.
    
    Args:
        query: The search query/question about compliance requirements
        n_results: Number of relevant chunks to retrieve (default: 5)
        
    Returns:
        A formatted string containing relevant compliance information
    """
    try:
        if not query or not query.strip():
            return "No search query provided. Please provide a query about compliance requirements."
        
        rag_service = get_rag_service()
        if not rag_service:
            logger.warning("RAG service not available when compliance_rag_search was called")
            return "RAG service is not available. The compliance document search feature is not initialized. " \
                   "This is not critical - you can proceed without it or try again later."
        
        # Retrieve relevant chunks
        try:
            chunks = rag_service.retrieve_relevant_chunks(query, n_results=n_results)
        except Exception as retrieval_error:
            logger.error(f"Error retrieving chunks from RAG: {retrieval_error}", exc_info=True)
            return f"Error retrieving compliance information: {str(retrieval_error)}. " \
                   "You can continue without RAG results or try again."
        
        if not chunks:
            return f"No relevant compliance information found for query: '{query[:100]}'. " \
                   f"This may mean no compliance documents have been uploaded yet, or the query doesn't match any documents. " \
                   "You can continue with general knowledge."
        
        # Format the results
        result_parts = [
            f"Found {len(chunks)} relevant sections from compliance documents for query '{query[:100]}':\n"
        ]
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('text', '')
            doc_name = chunk.get('document_name', 'Unknown Document')
            result_parts.append(
                f"\n--- Section {i} (from: {doc_name}) ---\n"
                f"{chunk_text}\n"
            )
        
        result = "\n".join(result_parts)
        logger.info(f"RAG search successful: {len(chunks)} chunks returned for query '{query[:50]}...'")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in compliance_rag_search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error searching compliance documents: {str(e)}. " \
               "This is not critical - you can proceed without RAG results."


# Create a tool wrapper compatible with Google ADK
def create_rag_tool():
    """Create a tool instance for Google ADK agents."""
    try:
        from google.adk.tools import Tool
        
        try:
            rag_tool = Tool(
                name="compliance_rag_search",
                description=(
                    "Search through uploaded compliance documents (FDA, ISO, IEC, HIPAA, etc.) "
                    "to find relevant regulatory information. Use this when you need specific "
                    "compliance requirements from your document library rather than general web search. "
                    "If no documents are uploaded or the search fails, the tool will return a message "
                    "indicating this, but the agent can continue working without RAG results."
                ),
                func=compliance_rag_search
            )
            logger.info("RAG tool created successfully for ADK agents")
            return rag_tool
        except Exception as tool_error:
            logger.error(f"Failed to create RAG tool: {tool_error}", exc_info=True)
            return None
            
    except ImportError as import_error:
        logger.warning(f"Google ADK tools not available: {import_error}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating RAG tool: {e}", exc_info=True)
        return None

