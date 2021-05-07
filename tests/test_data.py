import networkx as nx
import numpy as np
import pytest 

@pytest.fixture
def create_test_data():
    G_undir = nx.karate_club_graph()
    G = nx.Graph()
    G.add_edge('a', 'b', weight=0.6)
    G.add_edge('a', 'c', weight=0.2)
    G.add_edge('c', 'd', weight=0.1)
    G.add_edge('c', 'e', weight=0.7)
    G.add_edge('c', 'f', weight=0.9)
    G.add_edge('a', 'd', weight=0.3)

    return G,G_undir
