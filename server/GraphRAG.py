import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx
from langchain.chains import GraphQAChain
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

class GraphRAG:

    def __init__(self, llm, logger):
        self.logger = logger

        """Initialize the GraphRAG instance with required models and configurations."""
        self.llm = llm
        self.graph_transformer = self.initialize_graph_transformer()
        self.graph_ask_chain = self.initialize_need_graph_ask_chain()

    def initialize_graph_transformer(self):
        return LLMGraphTransformer(llm=self.llm)

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
    
    def needs_graph(self, user_query):
        return True
        # response = self.graph_ask_chain.invoke(user_query)
        # return True if response == 'yes' else False
    
    def build_graph_on_docs(self, context):
        documents = [Document(page_content=context)]
        return self.graph_transformer.convert_to_graph_documents(documents)