import keys
import utils
import parse_n_chunk
import embed
import search_n_retrieve
import bs4
# from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
from langchain_milvus import Milvus
from langchain_core.runnables import chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from typing import List
import nest_asyncio
from pymilvus import Collection, connections, utility

# Instiating a model
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


local_llm = 'llama3'
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# THIS WHOLE PORTION IS FOR INDEXING

def index_document(input_file, input_folder, parsed_folder, parser_choice):

    file_dict = utils.check_endpoint_for_file(input_file, input_folder, parsed_folder)
    
    file_base_name = list(file_dict.keys())[0]
    
    # Value is a tuple, { file : (.../data/file.pdf, .../parsed_data/file_parsed.txt) }
    input_file_path = file_dict.get(file_base_name, "default")[0]
    parsed_file_path = file_dict.get(file_base_name, "default")[1]

    # Check if document already parsed, if yes, skip, if no, parse
    try:
        with open(parsed_file_path, "r") as f:
            content = f.read().strip()

        if content:
            # If file has content, load from the file, # Future update, change this to directory, each pdf will have own pdf, to cache the parsed data, should pass a directoy path instead of file content
            pdf_docs = parse_n_chunk.load_parsed_docs(parser_name=parser_choice, input_file_content=content)

        else:
            # If file is empty, parse PDFs
            pdf_docs  = parse_n_chunk.intialise_n_parse(parser_choice, input_file_path, parsed_file_path)

    except FileNotFoundError:
        # If file doesn't exist, parse PDFs
        pdf_docs  = parse_n_chunk.intialise_n_parse(parser_choice, input_file_path, parsed_file_path)
        
    nodes = parse_n_chunk.chunk_parsed_markdown(documents=pdf_docs)

    return [file_base_name,nodes]

# # Print debug information
# print("Number of documents:", len(pdf_docs))
# if pdf_docs:
#     print("First document preview:", pdf_docs[0].get_content())

# # This is the websites that you used to retrieve info from
# urls = [
#     "https://www.ai-jason.com/learning-ai/how-to-reduce-llm-cost",
#     "https://www.ai-jason.com/learning-ai/gpt5-llm",
#     "https://www.ai-jason.com/learning-ai/how-to-build-ai-agent-tutorial-3"
# ]

# # Using WebBaseLoader to split doc
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# THIS PART HANDLES CHUNKING OF DATA

# Convert llama_index documents to LangChain Documents
# langchain_docs = []
# for doc in pdf_docs:
#     # Convert metadata and content
#     metadata = doc.metadata or {}
#     # # Clean metadata to ensure only simple types, this is only for ChromaDB, bc it obly supports these meta data formats
#     # clean_metadata = {k: v for k, v in metadata.items() 
#     #                   if isinstance(v, (str, int, float, bool))}
    
#     # Create LangChain Document
#     langchain_doc = Document(
#         page_content=doc.get_content(),  # or doc.text if get_content() doesn't work
#         metadata=metadata
#     )
#     langchain_docs.append(langchain_doc)

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=250, 
#     chunk_overlap=50  # Added a small overlap for context preservation
# )
# split_docs = text_splitter.split_documents(langchain_docs)

# THIS PORTION IS ON INITIALISING AND POPULATNG THE VECTOR DB

def setup_vector_store():
    # Initialise Milvus DB
    connection_args={"host": "127.0.0.1", "port": "19530"}
    embeddings = embed.CustomEmbedder()

    # Initialise Milvus 
    vectorstore = Milvus(
        connection_args=connection_args,
        embedding_function=embeddings, # Might what to switch to node embedding instead of text
        drop_old=True
    )

    print(vectorstore.client.list_collections())

    return vectorstore

def setup_collection(collection_name, vectorstore, documents, ids):

    embeddings=embed.CustomEmbedder()


    connection_args = setup_milvus_connection()

    # connection_args={"host": "127.0.0.1", "port": "19530"}

    index_params = {
            'metric_type': 'COSINE',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 128}
        }

    search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 12},  # Number of clusters to probe
        }

    # This is to drop the old collection, and create a new collection. If set to false, creating set_up col will add to current col is col_name used is the same.
    drop_old = True

    print("adding document chunks to db")

    vectorstore.from_documents(documents=documents, embedding=embeddings, collection_name=collection_name, index_params=index_params,search_params=search_params, ids=ids, connection_args=connection_args, drop_old=drop_old)

    print("finished adding document chunks to db")

def setup_milvus_connection():
    """Establish connection to Milvus"""
    connection_args = {
        "host": "127.0.0.1", 
        "port": "19530"
    }
    
    # Connect to Milvus
    connections.connect(
        alias="default",
        **connection_args
    )
    
    return connection_args

def load_collections(vectorstore):
    """
    Load and return all collections from Milvus instance.
    Returns:
        list: List of Collection objects.
    """
    collections = {}
    embedding_function=embed.CustomEmbedder()

    # First establish connection
    connection_args = setup_milvus_connection()

    # Get list of all collection names
    collection_names = vectorstore.client.list_collections()
        
    # Load each collection
    for col_name in collection_names:
        # collection = Collection(col_name)
        # collection.load()

        collection = Milvus(collection_name=col_name, connection_args=connection_args, embedding_function=embedding_function)

        if embedding_function:
            collection.embedding_function = embedding_function

        collections[col_name] = collection
    
    return collections


def add_collection(vectorstore, doc_name, actual_nodes):
    """
    Function that takes in text nodes, converts them into vectorBD friendly langchain docs, 
    chunks them to required size, and creates an id for each chunk,
    then sets up a collection

    Args: vectorstore instance, document or collection name, nodes to add to new collection
    
    """
    print("Collection name: "+ str(doc_name))
    print("Number of nodes: "+ str(len(actual_nodes)))

    # Convert nodes into Langchain documents for VectorDB Storage
    langchain_docs = embed.cnvert_nodes_to_langchain_docs(actual_nodes)
    print("Number of langchain_docs: "+ str(len(langchain_docs)))

    # Chunk Langchain documents to required size
    chunked_nodes = parse_n_chunk.chunker(langchain_docs)

    ids = [str(i) for i in range(len(chunked_nodes))]
    print("Number of chunks: "+ str(len(chunked_nodes)))

    setup_collection(collection_name=doc_name, vectorstore=vectorstore, documents=chunked_nodes, ids=ids)

@chain
def retriever(querystate) -> list[Document]:

    query = str(querystate["query"])
    vectorstore = querystate["vectorstore"]


    # Load the vector with the collection used for retrieval
    collections = load_collections(vectorstore=vectorstore)

    multi_retriever = search_n_retrieve.MultiCollectionRetriever(collections)
    
    # Merged retrieval
    docs = multi_retriever.retrieve_merged(query, k=4)

    print(docs)
    
    # # Debugging to print the current laoded collection
    # print(vectorstore.client.get_collection_stats("rag_milvus"))

    # # Custom search parameters for Milvus
    # docs, scores = zip(*vectorstore.similarity_search_with_score(query=query, param=search_params, k=4))
    # for doc, score in zip(docs, scores):
    #     doc.metadata["score"] = score
    #     # print(f"Similarity Score: {score}")
    return docs

def index_and_embed_cur_docs(input_folder="./data", parsed_folder="./parsed_data", parser_choice="Marker"):

    vectorstore = setup_vector_store()
    cur_collections_names = vectorstore.client.list_collections()

    keys.os.makedirs(parsed_folder, exist_ok=True)

    # If document has been removed from corpus, delete respective collection
    input_files = keys.os.listdir(input_folder)

    input_collection_names = []

    for input_file_name in input_files:

        # Skip hidden files
        if input_file_name.startswith('.'):  # Skip hidden files like .DS_Store
            continue
        else:
            input_file_base_name = keys.os.path.splitext(input_file_name)[0]
            input_collection_names.append(input_file_base_name)
        
        # Skip non-file entries
        data_file_path = keys.os.path.join(input_folder, input_file_name)
        if not keys.os.path.isfile(data_file_path):
            continue

        # If collection already exists for current file, dont re-add and just skip
        if input_file_base_name not in cur_collections_names:
            add_document_to_DB(vectorstore, input_file_name, input_folder, parsed_folder, parser_choice)

    if set(cur_collections_names).issubset(input_collection_names):
        return vectorstore
    
    for name in cur_collections_names:
        if name not in input_collection_names:
            vectorstore.client.drop_collection(name)
            print("Droppped "+ name)
    
    return vectorstore

def add_document_to_DB(vectorstore, input_file_name, input_folder="./data", parsed_folder="./parsed_data", parser_choice="Marker"):
    """
    Function to add a new document to the Vector DB

    Args: vectorestore instance, input_file_name, input_folder, parsed_folder, parser_choice
    
    """

    indexed_doc = index_document(input_file_name, input_folder, parsed_folder, parser_choice)

    # add collection for each doc
    doc_base_name = indexed_doc[0]
    actual_nodes = indexed_doc[1]

    add_collection(vectorstore, doc_base_name, actual_nodes)

    return vectorstore

# THIS PORTION IS ON RETRIEVAL

# THIS PORTION IS ON RETRIEVAL GRADING
prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n 
    It. does not need to be a stringent test. \n 
    The goal is to filter out erroneous retrievals. \n
    Give a binary score 'YES' or 'NO' score to indicate whether the document is relevant to the question.\n
    Provide the binary score as value to a single key 'score' in JSON format.
    Do not provide any preamble or explanation. 

    <|eot_id|><|start_header_id|>user<|end_header_id|>s
    Here is the retrieved document: \n\n {documents} \n\n
    Here is the user question: {question} \n 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "documents"],
)

retrieval_grader = prompt | llm | JsonOutputParser()


# THIS PORTION IS ON ANSWER GENERATION
prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Provide your response as value to a single key 'generation' in JSON format.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {documents} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "documents"],
)

# # Postprocessing of generated reply
# def format_docs(docs):
#     return

rag_chain = prompt | llm | StrOutputParser()


# THIS PORTION IS ON HALLUCINATION GRADING
prompt = PromptTemplate(
    template="""     
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    system You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
    Only give a binary score 'YES' or 'NO' score to indicate whether the answer is grounded in / supported by a set of facts. 
    Provide the binary score as value to a single key 'score' in JSON format.
    Do not provide any preamble or explanation. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["documents", "generation"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

# THIS PORTION IS ON ANSWER GRADING
prompt = PromptTemplate(
    template="""     
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is useful to resolve a question. 
    Give a binary score 'YES' or 'NO' to indicate whether the answer is useful to resolve a question. 
    Provide the binary score as value to a single key 'score' in JSON format.
    Do not provide any preamble or explanation. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

# THIS IS FOR WEB SEARCHING 
# Uses Tavily API
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

# LANG GRAPH
# this defines the states for the llm workflow
from typing_extensions import TypedDict
from typing import List
from langgraph.errors import GraphRecursionError

class GraphState(TypedDict):
    """
    Represents the state of our graph. 
    What kind of values you want to share across all the differnt nodes of the graph

    Attributes:
        question: question
        documents: list of documents
        generation: LLM generation
        search: whether to add search
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]

# This is the different nodes in our GRAPH 
def retrieve(state, vectorstore):
    """
    Retrieve different documents from vectorstore

    Args/Input:
        state (which is a dict): The current graph state

    Returns/Output:
        state (which is a dict): New key added to state, documents, which is retrievd from store 
    """
    # print("=== RETRIEVED ===")

    # Get the current question in graph state
    question = state["question"]

    # Retrieval
    documents = retriever.invoke({"query": question, "vectorstore": vectorstore})

    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Grade retrieved documents, if at least one relevant do not search web, if all not relevant, search web

    Args/Input:
        state (which is a dict): The current graph state

    Returns/Output:
        state (which is a dict): New key added to state, web_search, which decides whether to search web based on the grade of teh retrieved documents
    """

    # print("=== CHECKING RELEVANCE ===")

    # Get the current question in graph state
    question = state["question"]
    documents = state["documents"]

    # Score each document
    filtered_docs = []
    web_search = 'NO'
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "documents": doc})
        grade = score["score"]
        
        if grade.upper() == 'YES':
            #  print("= RELEVANT =")
             filtered_docs.append(doc)
        else:
            # print("= NOT RELEVANT =")
            continue
    
    if (len(filtered_docs) == 0): web_search = 'YES'

    return {"question": question, "documents": filtered_docs, "web_search": web_search}

def generate(state):
    """
    Generate answer based on retrieved documents

    Args/Input:
        state (which is a dict): The current graph state

    Returns/Output:
        state (which is a dict): New key added to state, generation, which is the answer generated by llm
    """

    # print("=== GENERATED ===")
    
    question = state["question"]
    documents = state["documents"]

    try:
        generated_ans = rag_chain.invoke({"question": question, "documents": documents}, {"recursion_limit": 4})
    except GraphRecursionError:
        print("Recursion Error")
        return

    return {"question": question, "documents": documents, "web_search": web_search, "generation": generated_ans}


def web_search(state):
    """
    Web search based on question 

    Args/Input:
        state (which is a dict): The current graph state

    Returns/Output:
        state (which is a dict): Appended web resuls, to value of documents
    """

    # print("=== WEB SEARCH ===")

    question = state["question"]
    documents = state["documents"]

    # Search the web 
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join(d["content"] for d in docs)
    web_results = Document(page_content=web_results)

    if not documents:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

def decide_to_generate(state):
    """
    Determines whether to generate and answer or continue searching the web. This is the conditional node

    Args/Input:
        state (which is a dict): The current graph state

    Returns/Output:
        state (which is a dict): Binary decision for next node
    """

    # print("=== ASSESS NEED TO SEARCH ===")

    web_search = state["web_search"]

    if(web_search == 'YES'):
        # print("= ALL DOCUMENTS ARE IRRELEVANT =")
        return "websearch"

    # Have at least 1 document
    else:
        # print("= AT LEAST 1 DOCUMENT RELEVANT =")
        return "generate"

def check_hallucination(state):
    """
    Determines whether LLM is hallucinating

    Args/Input:
        state (which is a dict): The current graph state

    Returns/Output:
        state (which is a dict): Binary decision on whether hallucinating
    """

    # print("=== CHECKING FOR HALLULLU ===")

    question = state["question"]
    documents = state["documents"]
    generated_ans = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generated_ans})
    grade = score['score']

    if grade.upper() == 'YES': 
        # print("=== NO HALLULLU ===")

        # print("=== CHECKING FOR ATQ ===")
        score = answer_grader.invoke({"question": question, "generation": generated_ans})
        grade = score['score']
        if grade.upper() == 'YES':
            # print("=== YES ATQ ===")
            return "useful"
        else:
            # print("=== NO ATQ ===")
            return "not useful"

    else:
        # print("=== YES HALLULLU, RETRY ===")
        return "not supported"
    

# THIS IS FOR GRAPH CREATION 
from langgraph.graph import END, StateGraph
from functools import partial

def create_graph(vectorstore):
    """
    Creates the StateGraph and workflow

    Returns/Output:
        Workflow: Stategraph object representing the workflow 
    """
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("retrieve", partial(retrieve, vectorstore=vectorstore))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("websearch", web_search)
    workflow.add_node("generate", generate)

    # Build the graph
    workflow.set_entry_point("retrieve")
    # Add an edge between retrieve and grade_doc node
    workflow.add_edge("retrieve", "grade_documents")
    # Add conditional edge btw grade_doc and decide_to_gen
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate, 
        {
            "websearch": "websearch",
            "generate": "generate",
        }
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        check_hallucination, 
        {
            "not supported": "generate",
            "not useful": "websearch",
            "useful": END,
        }
    )

    return workflow

# from IPython.display import Image, display

if __name__ == "__main__":

    vectorstore = index_and_embed_cur_docs()

    # Compile workflow (assuming this is already done)
    workflow = create_graph(vectorstore)
    app = workflow.compile()

    # display(Image(app.get_graph().draw_mermaid_png()))

    inputs = {"question": "Broadly describe the desgin of the system used to improve scalibility and accuracy of the robot in the RL at Scale experiment."}

    # Generate answer
    for output in app.stream(inputs):
        for key, value in output.items():
            generated_answer = value.get("generation", "")
        
    print(f"{inputs['question']}")
    print(f"Generated: {generated_answer}")
    