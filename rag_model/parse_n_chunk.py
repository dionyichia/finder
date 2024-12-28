import os
from . import utils
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# import pymupdf # imports the pymupdf library

# PARSING ================================

def intialise_n_parse(parser_name: str, input_file_path, parsed_file_path):

    if parser_name.lower() == "llamaparse":

        # Initialize the parser
        parser = LlamaParse(
            result_type="markdown",  # "markdown" and "text" are available
            verbose=True,
            num_workers=4,
            split_by_page=0,
        )

        file_extractor = {".pdf": parser}

        pdf_docs = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

        write_to_parsed_f(parser_name=parser_name, parsed_file_path=parsed_file_path, content=pdf_docs)

    elif parser_name.lower() == "marker":

        pdf_docs = []

        parser = PdfConverter(
            artifact_dict=create_model_dict(),
        )

        pdf_content = parser(input_file_path)
        pdf_docs.append(pdf_content)
    
        # Update in future to add more pdfs at a time
        # For pdf_doc in pdf_docs: then load into separate file paths
        write_to_parsed_f(parser_name=parser_name, parsed_file_path=parsed_file_path, content=pdf_docs[0])


    # Use a model to parse images into markdwon
    elif parser_name.lower() == "model":        
         pass

    else:
        print("Error, unknown parser selected.")
        pdf_docs = None
    
    return pdf_docs

def load_parsed_docs(parser_name: str, input_file_content):

    json_data = utils.json.loads(input_file_content)
        
    if parser_name.lower() == "llamaparse":
        # Deserialize the JSON to reconstruct Document
        pdf_docs = [utils.dict_to_document(json_data)]

    elif parser_name.lower() == "marker":
        pdf_docs = [utils.decode_markdown_output(json_data)]

    else:
        print("Error, unknown parser selected.")
        pdf_docs = None

    return pdf_docs

def write_to_parsed_f(parser_name: str, parsed_file_path, content):

    # Save the document's full details to the .txt file
    with open(parsed_file_path, "w") as f:
        f.seek(0)

        # TO CHANGE THIS TO SUPPORT MULTI PDFS, NEED TO CHANGE THE WAY THE IT IS LOADED ADN RETRIEVED FROM THE .TXT FILE, CURRENTLY ONLY 1 DOC THERFORE PDF_DOCS[0] WORKS, WILL NOT WORK FOR MUTLI DOC
        if parser_name.lower() == "llamaparse":
        # Deserialize the JSON to reconstruct Document
            utils.json.dump(utils.document_to_dict(content), fp=f, indent=2)

        elif parser_name.lower() == "marker":
            utils.json.dump(content, fp=f, cls=utils.MarkdownOutputEncoder, indent=2)

        else:
            print("Error, unknown parser selected.")       

        f.truncate()



# CHUNKING ================================

# Headers = "\n#"
# Page = "\n---\n#"
# 

# FOR NOW ONLY HANDLE MARKDOWN TEXT, FUTURE UPDATES, HANDLE IMAGES, HANDLE TABLES separately
def chunk_parsed_markdown(documents):

    # Convert Markdown Output into Tree structure
    markdown_text = documents[0].markdown
    tree = utils.markdown_to_tree(markdown_text)
    # print(utils.json.dumps(tree, indent=2))

    # Recusively do the following: One pass through tree
    
    # Account for paragraphs that are split in btw nodes , use llm w/ prompt template to check +2 -2 paras from current node to decide whether shld be joined tgt
    # Reformat Nodes
    # Ignore structure and append all nodes into a list 
    def flatten_tree(node):
        """
        Flatten the Markdown tree into a list of dictionaries, 
        
        :param node: Root node of the Markdown tree
        :param current_path: Current header path (for internal recursion)
        :return: List of dictionaries containing all nodes
        """
        # Initialize the list to store flattened nodes
        flattened_nodes = []

        # Handle the root node separately
        if node['heading']['title'] == "Document Root":
            # Recursively process children of the root
            for child in node.get('children', []):
                flattened_nodes.extend(flatten_tree(child))
        else:
            # For non-root nodes, create a dictionary representation
            # Create a node dictionary with all relevant information
            node_dict = {
                'metadata': {
                    'title': node['heading']['title'],
                    'depth': node['heading']['depth'],
                    'header_path':  node['heading']['header_path'],
                },
                'text': node.get('content', ''),
                # Optionally add more metadata as needed
            }
            
            # Add the current node to the flattened list
            flattened_nodes.append(node_dict)
            
            # Recursively process children
            for child in node.get('children', []):
                flattened_nodes.extend(flatten_tree(child))
        
        return flattened_nodes
    
    list_of_nodes = flatten_tree(tree)

    return list_of_nodes

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker

def chunker(langchain_docs):

    def chunk_by_size(langchain_docs):
        chunker = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=1024,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

        return chunker.split_documents(langchain_docs)
    
    return chunk_by_size(langchain_docs)


    # Split each node in tree semantically

    # TO USE LANGCHAIN TEMPORARILY, FURTHER UPDATES TO CREATE OWN SPLITTER
    def chunk_by_semantics():

        # chunker = SemanticChunker(
        #     OpenAIEmbeddings(), breakpoint_threshold_type="gradient"
        # )
        
        pass



    # Return list of chunks, content : "...", meta: "..."

    # # Split whole
    # node_parser = MarkdownNodeParser(llm=None)
    # nodes = node_parser.get_nodes_from_documents(documents=documents)

    # return nodes