import json
from sentence_transformers import SentenceTransformer

def create_rich_embedding_text(node):
    """
    Create a comprehensive text representation that includes metadata and structure
    
    Args:
        node: A node object from llama_index
    
    Returns:
        str: A rich text representation of the node
    """
    # Combine text content with metadata in a structured way
    rich_text = f"""
    Content: {node.text}
    
    Metadata:
    {json.dumps(node.metadata, indent=2)}
    
    Additional Context:
    - Node Type: {type(node).__name__}
    - Node Depth: {node.metadata.get('depth', 'Unknown')}
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

# # More advanced multi-field embedding
class MultiFieldEmbedder:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, node):
        # Combine different aspects of the node
        text_features = node.text
        metadata_features = json.dumps(node.metadata)
        
        # Concatenate or use a more sophisticated combination
        combined_features = f"{text_features}\n\nMetadata:\n{metadata_features}"
        
        return self.model.encode(combined_features)

# # Usage
# node_embeddings = [embedder.embed_node(node) for node in nodes]

