import os
import networkx as nx
import json_repair
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain.graphs.graph_document import GraphDocument, Node, Relationship

class GraphRAG:

    def __init__(self, llm, logger):
        self.logger = logger

        """Initialize the GraphRAG instance with required models and configurations."""
        self.llm = llm
        self.graph_transformer = self.initialize_graph_transformer()
        self.graph_ask_chain = self.initialize_need_graph_ask_chain()

    # Initialize LLMGraphTransformer
    def initialize_graph_transformer(self):
        return LLMGraphTransformer(llm=self.llm)

    # Build chain for determining whether or not user needs graph
    def initialize_need_graph_ask_chain(self):
        self.logger.info('CREATED GRAPH ASK CHAIN')
        """Create the chain to ask if user is requesting a graph"""
        graph_needed_thread = [
            ('system', os.getenv('needs_graph_ask_instruction')),
            ('human', os.getenv('needs_graph_ask_question'))
        ]
        graph_needed_prompt = ChatPromptTemplate.from_messages(graph_needed_thread)
        graph_needed_llm_chain = graph_needed_prompt | self.llm
        return {"question": RunnablePassthrough()} | graph_needed_llm_chain
    
    # Build chain to normalize graph using LLM.
    def build_normalize_graph_chain(self):
        normalize_graph_thread = [
            ('system', os.getenv('needs_graph_ask_instruction')),
            ('human', os.getenv('needs_graph_ask_question'))
        ]
        normalize_graph_prompt = ChatPromptTemplate.from_messages(normalize_graph_thread)
        normalize_graph_llm_chain = normalize_graph_prompt | self.llm
        return {"network": RunnablePassthrough()} | normalize_graph_llm_chain

    # Check if user needs graph based on question
    def needs_graph(self, user_query):
        response = self.graph_ask_chain.invoke(user_query)
        return True if response == 'yes' else False
    
    # Build graph from context
    def build_graph_on_docs(self, context):
        documents = [Document(page_content=context)]
        return self.graph_transformer.convert_to_graph_documents(documents)
    
    # Build GraphDocument based on LLM JSON output
    def build_graph_document_from_llm(llm_out, context):
        graph_json = json_repair.loads(llm_out)
        nodes = [Node(id=node['id'], type=node['type']) for node in graph_json['nodes']]
        relationships = [Relationship(source=relationship['source'], target=relationship['target'], type=relationship['type']) for relationship in graph_json['relationships']]
        src_doc = Document(page_content=context)
        graph_doc = GraphDocument(nodes=nodes, relationships=relationships, source=src_doc)
        return graph_doc 
