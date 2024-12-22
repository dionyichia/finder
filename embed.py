import json
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document

def cnvert_nodes_to_langchain_docs(nodes):
    langchain_docs = [
            Document(
                page_content=create_rich_embedding_text(node), 
                metadata={}
            ) for node in nodes
        ]

    return langchain_docs

def create_rich_embedding_text(node):
    """
    Create a comprehensive text representation that includes metadata and structure
    
    Args:
        node: A node object
    
    Returns:
        str: A rich text representation of the node
    """
    # Combine text content with metadata in a structured way
    rich_text = f"""
    Content: {node.get("text", '')}
    
    Metadata: {json.dumps(node.get("metadata", ''), indent=2)}
    
    Node Type: {type(node).__name__}
    """
    return rich_text

def embed_documents(rich_texts, embedding_model):
    """
    Embed rich text representations of documents
    
    Args:
        rich_texts (list): List of rich text representations
        embeddings: Embedding model
    
    Returns:
        list: List of embeddings
    """
    return embedding_model.embed_documents(rich_texts)
# Use this rich text for embedding


# from sentence_transformers import SentenceTransformer

class CustomEmbedder:
    """
    params: Chunked LangChain Documents
    
    return: List of Embeddings
    """

    def __init__(self, model_name='Alibaba-NLP/gte-large-en-v1.5'):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using specified embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        embeddings = [self.model.encode(text) for text in texts]

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
    
        return self.embed_documents([text])[0]
    

# # Usage
# node_embeddings = [embedder.embed_node(node) for node in nodes]

