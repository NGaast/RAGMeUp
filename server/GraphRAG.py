import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx
from langchain.chains import GraphQAChain
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph


class GraphRAG:

    def __init__(self):
        """Initialize the GraphRAG instance with required models and configurations."""
        self.llm = OllamaLLM(model='llama3.1')

        self.text = """
        Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
        She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
        Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
        She was, in 1906, the first woman to become a professor at the University of Paris. 
        """

    def load_text(self):
        self.documents = [Document(page_content=self.text)]
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        graph_documents = llm_transformer.convert_to_graph_documents(self.documents)

    def build_graph(self):
        llm_transformer_filtered = LLMGraphTransformer(
        llm=self.llm,
        allowed_nodes=["Person", "Country", "Organization"],
        allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
        )

        graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
            self.documents
        )
        self.graph = NetworkxEntityGraph()

        # Add nodes to the graph
        for node in graph_documents_filtered[0].nodes:
            self.graph.add_node(node.id)

        # Add edges to the graph
        for edge in graph_documents_filtered[0].relationships:
            self.graph._graph.add_edge(
                    edge.source.id,
                    edge.target.id,
                    relation=edge.type,
                )
            
    def build_graph_chain(self):
        self.chain = GraphQAChain.from_llm(
                llm=self.llm, 
                graph=self.graph, 
                verbose=True
            )
        
    def test_query(self):
        question = """Who is Marie Curie?"""
        self.chain.run(question)