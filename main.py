import keys
import bs4
# from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
from langchain_milvus import Milvus
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
import embed

# Instiating a model
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


local_llm = 'llama3'
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# THIS WHOLE PORTION IS FOR INDEXING

parser = LlamaParse(
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
    num_workers=4,
    split_by_page = 0,
)

def index_documents():
    file_extractor = {".pdf": parser}
    pdf_docs = SimpleDirectoryReader(
        "./data", file_extractor=file_extractor
    ).load_data()


    node_parser = MarkdownNodeParser(llm=None)
    nodes = node_parser.get_nodes_from_documents(documents=pdf_docs)
    return nodes

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
# Embedding parsed documents into Vector DB 
# embeddings = GPT4AllEmbeddings.embed_documents(nodes)


def setup_vector_store():
    # Initialise Milvus DB
    URI = "./vector_db.db"

    index_params = {
        'metric_type': 'COSINE',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 128}
    }

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 12},  # Number of clusters to probe
    }

    vector_store = Milvus (
        connection_args={"host": "127.0.0.1", "port": "19530"},
        embedding_function=GPT4AllEmbeddings(), # Might what to switch to node embedding instead of text
        index_params=index_params,
        search_params=search_params,
    )

    return vector_store

# Inserting vectors in Vector DB 
# Prepare ids of each vector

def index_and_embed():

    nodes = index_documents()
    vector_store = setup_vector_store()
    # node_embeddings = [embed.create_rich_embedding_text(node) for node in nodes]

    # Convert nodes into rich nodes and into Langchain Documents for storage
    rich_documents = [
        Document(
            page_content=embed.create_rich_embedding_text(node), 
            metadata=node.metadata if hasattr(node, 'metadata') else {}
        ) for node in nodes
    ]

    ids = [str(i) for i in range(len(rich_documents))]

    # Insert documents with their embeddings
    insert_result = vector_store.add_documents(documents=rich_documents, ids=ids, collection_name="rag_milvus")
    print("Insert Result: ", insert_result)

# Overloading above default retriever function, to add sim scores in metadata and print scores
from langchain_core.runnables import chain
@chain
def retriever(query: str) -> list[Document]:
    vector_store = setup_vector_store()

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 12},
    }

    # Custom search parameters for Milvus
    docs, scores = zip(*vector_store.similarity_search_with_score(query=query, param=search_params))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
        print(score)
    return docs

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
def retrieve(state):
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
    documents = retriever.invoke(question)

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

    generated_ans = rag_chain.invoke({"question": question, "documents": documents})

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

def create_graph():
    """
    Creates the StateGraph and workflow

    Returns/Output:
        Workflow: Stategraph object representing the workflow 
    """
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("retrieve", retrieve)
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

if __name__ == "__main__":

    index_and_embed()

    # Compile workflow (assuming this is already done)
    workflow = create_graph()
    app = workflow.compile()

    inputs = {"question": "How many hours of robotic experience were gathered for training the RL@Scale system?"}

     # Generate answer
    for output in app.stream(inputs):
        for key, value in output.items():
            generated_answer = value.get("generation", "")
        
    print(f"{inputs['question']}")
    print(f"Generated: {generated_answer}")
    