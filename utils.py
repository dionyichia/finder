import json
from llama_index.core.schema import Document

def document_to_dict(doc):
    """
    Convert a Document object to a dictionary for JSON serialization
    """
    return {
        'id_': doc.id_,
        'embedding': doc.embedding,
        'metadata': doc.metadata,
        'excluded_embed_metadata_keys': doc.excluded_embed_metadata_keys,
        'excluded_llm_metadata_keys': doc.excluded_llm_metadata_keys,
        'relationships': doc.relationships,
        'metadata_template': doc.metadata_template,
        'metadata_separator': doc.metadata_separator,
        'text': doc.text,
        'mimetype': doc.mimetype,
        'start_char_idx': doc.start_char_idx,
        'end_char_idx': doc.end_char_idx,
        'metadata_seperator': doc.metadata_seperator,
        'text_template': doc.text_template
    }

def dict_to_document(doc_dict):
    """
    Reconstruct a Document object from a dictionary
    """
    return Document(
        id_=doc_dict['id_'],
        embedding=doc_dict['embedding'],
        metadata=doc_dict['metadata'],
        excluded_embed_metadata_keys=doc_dict['excluded_embed_metadata_keys'],
        excluded_llm_metadata_keys=doc_dict['excluded_llm_metadata_keys'],
        relationships=doc_dict['relationships'],
        metadata_template=doc_dict['metadata_template'],
        metadata_separator=doc_dict['metadata_separator'],
        text=doc_dict['text'],
        mimetype=doc_dict['mimetype'],
        start_char_idx=doc_dict['start_char_idx'],
        end_char_idx=doc_dict['end_char_idx'],
        metadata_seperator=doc_dict['metadata_seperator'],
        text_template=doc_dict['text_template']
    )