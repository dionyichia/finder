import json
import os
from llama_index.core.schema import Document

# PARSING ================================

def check_endpoint_for_file(input_file, input_folder, parsed_folder):
    """
    For the file, check if a respective parsed file exists in the "parsed_folder". If not, create an endpoint file in the "parsed_folder".

    Args:
        data_folder (str): Path to the input folder containing files to check.
        parsed_folder (str): Path to the folder where parsed files should exist.

    Returns:
        None
    """
    # # Ensure the parsed_folder exists
    # os.makedirs(parsed_folder, exist_ok=True)

    # for input_file_name in os.listdir(data_folder):

        # Skip hidden files
        # if input_file_name.startswith('.'):  # Skip hidden files like .DS_Store
        #     continue
        
        # # Skip non-file entries
        # data_file_path = os.path.join(data_folder, input_file_name)
        # if not os.path.isfile(data_file_path):
        #     continue

    file_dict = {}

    # Determine the corresponding parsed file name
    data_file_path = os.path.join(input_folder, input_file)
    base_name, _ = os.path.splitext(input_file)
    parsed_file_name = f"{base_name}_parsed.txt"
    parsed_file_path = os.path.join(parsed_folder, parsed_file_name)

    # Check if the parsed file exists
    if not os.path.exists(parsed_file_path):
        # Create an empty endpoint file if it doesn't exist
        with open(parsed_file_path, 'w') as f:
            f.write("")
        print(f"Created missing parsed file: {parsed_file_name}")

    file_dict[base_name] = (data_file_path, parsed_file_path)

    return file_dict
        

# This is for LlamaParse
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

# This is for Marker
import base64
from typing import Dict, List
from PIL import Image
import io

class MarkdownOutput:
    def __init__(self, markdown: str = '', images: Dict[str, any] = None, metadata: Dict[str, any] = None):
        self.markdown = markdown
        self.images = images or {}
        self.metadata = metadata or {}

class MarkdownOutputEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            # Convert the object to a dictionary representation
            return self._convert_to_dict(obj)
        
        return super().default(obj)
    
    def _convert_to_dict(self, obj):
        """
        Recursively convert an object to a JSON-serializable dictionary.
        
        Args:
            obj: The object to convert
        
        Returns:
            A JSON-serializable representation of the object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_dict(v) for k, v in obj.items()}
        
        if hasattr(obj, 'images') and obj.images:
            # Special handling for images
            encoded_images = {}
            for key, image in obj.images.items():
                try:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    encoded_images[key] = {
                        "base64": base64.b64encode(buffered.getvalue()).decode('utf-8'),
                        "format": image.format,
                        "mode": image.mode,
                        "size": image.size
                    }
                except Exception as e:
                    print(f"Warning: Could not encode image {key}: {e}")
            
            # Create a dictionary representation of the object
            obj_dict = {
                "markdown": getattr(obj, 'markdown', ''),
                "images": encoded_images,
                "metadata": getattr(obj, 'metadata', {})
            }
            return obj_dict
        
        # For other objects, convert to dictionary if possible
        if hasattr(obj, '__dict__'):
            return {k: self._convert_to_dict(v) for k, v in obj.__dict__.items()}
        
        return obj

def decode_markdown_output(json_data):
    """
    Decode a JSON representation of MarkdownOutput back to MarkdownOutput object
    """
    # Reconstruct images
    images = {}
    if 'images' in json_data:
        for key, img_data in json_data['images'].items():
            # Decode base64 back to PIL Image
            image_bytes = base64.b64decode(img_data['base64'])
            images[key] = Image.open(io.BytesIO(image_bytes))
    
    # Create a new MarkdownOutput-like object
    return MarkdownOutput(
        markdown=json_data['markdown'],
        images=images,
        metadata=json_data['metadata']
    )


# CHUNKING ================================

import commonmark

def markdown_to_tree(markdown_text):
    """
    Convert Markdown text to a nested tree structure using commonmark.
    
    :param markdown_text: String containing Markdown content
    :return: Dictionary representing the parsed Markdown tree
    """
    parser = commonmark.Parser()
    ast = parser.parse(markdown_text)
    
    def process_node(node, current_parent=None):
        """
        Recursively process AST nodes and convert to desired tree structure.
        
        :param node: Current AST node
        :param parent_type: Type of parent node
        :return: Processed node dictionary
        """

        # Root node
        root = {
            "heading": {
                "depth": 1,
                "title": "Document Root"
            },
            "content": None,
            "children": [],
        }

        # Stack to keep track of parent nodes
        parent_stack = [root]

        def add_node(node):

            # If is not a heading, add all content to pevious parent node, i.e associate each node with a heading
            if node.t != "heading":
                content = extract_content(node)
                if content:
                    # Append content to the most recent header's content
                    if parent_stack[-1].get("content") is None:
                        parent_stack[-1]["content"] = ""
                    parent_stack[-1]["content"] += content + "\n\n"
                return None

            # Find the correct parent based on heading depth
            while (len(parent_stack) > 1 and parent_stack[-1]["heading"]["depth"] >= node.level):
                parent_stack.pop()

            parent = parent_stack[-1]
            
            # Initialize the new header_node
            title = extract_text(node)
            new_node = {    
                "heading": {
                    "title": title,
                    "depth": node.level,
                    "header_path": add_header_paths(
                        title, 
                        parent.get('heading', '').get("header_path", '')),
                },
                "content": extract_content(node),
                "children": [],
            }

            parent_stack.append(new_node)

            # Add to parent's children
            parent_stack[-2]["children"].append(new_node)
            
            return new_node

         # Traverse the AST
        def traverse(current_node):
            if not current_node:
                return

            # Special handling for different node types
            if current_node.t in ["heading", "paragraph", "code_block", "list"]:
                add_node(current_node)

            # Recursively process children
            child = current_node.first_child
            while child:
                traverse(child)
                child = child.nxt

        # Start traversal from the document root
        traverse(ast)
        
        return root

    def extract_text(node):
        """
        Extract text from a node recursively.
        
        :param node: AST node
        :return: Extracted text string
        """
        if not node:
            return ""
        
        # Direct literal for simple nodes
        if hasattr(node, 'literal') and node.literal:
            return node.literal
        
        # Recursively extract text from child nodes
        text = ""
        child = node.first_child
        while child:
            if hasattr(child, 'literal') and child.literal:
                text += child.literal
            child = child.nxt
        
        return text.strip()
    
    def extract_content(node):
        """
        Extract full content from a node recursively.

        :param node: AST node
        :return: Extracted content string
        """
        if not node:
            return ""

        content = ""
        if hasattr(node, 'literal') and node.literal:
            content += node.literal

        child = node.first_child
        while child:
            content += extract_content(child)
            child = child.nxt

        return content.strip()

    # Find all path information to each block and add into respective dict
    def add_header_paths(current_header, current_path=None, separator='/'):
        """
        Recursively add header paths to each node in the tree.
        
        :param node: Current node in the tree
        :param current_path: Current header path (str of header titles), FURTHER UPDATES TO CONSIDER CHANGING TO LIST
        :return: Modified node with header_path added
        """
        if not current_header or current_header == "Document Root":
            return ""
        
        return f"{current_path}{separator}{current_header}".lstrip(separator)
            
        # # Recursively process children
        # for child in node.get('children', ''):
        #     # Pass the current node's header path as the base for its children
        #     add_header_paths(child, node['header_path'])

    # Process the entire document
    root = process_node(ast)

    return root


        
        # elif node.t == "list":
        #     current_item = node.first_child
        #     while current_item:
        #         item_content = extract_content(current_item)
        #         if item_content:
        #             new_node["children"].append({
        #                 "children": [],
        #                 "content": item_content,
        #                 "heading": None
        #             })
        #         current_item = current_item.nxt
        #     current_parent["children"].append(new_node)
