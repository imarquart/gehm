import os

os.chdir("d:\\Marquart\\Documents\\Python\\gehm\\")

from gehm.datasets.nx_datasets import *
from gehm.utils.config import process_config
from gehm.utils.file_helpers import check_create_folder
from gehm.agents.hsdne import hSDNEAgent

from gehm.model.positions import Disk2

def create_test_data():
    G_undir = nx.karate_club_graph()
    G = nx.Graph()
    G.add_edge("a", "b", weight=0.6)
    G.add_edge("a", "c", weight=0.2)
    G.add_edge("c", "d", weight=0.1)
    G.add_edge("c", "e", weight=0.7)
    G.add_edge("c", "f", weight=0.9)
    G.add_edge("a", "d", weight=0.3)
    hierarchy_dict={}
    hierarchy_dict["a"]={"hierarchy": 0}
    hierarchy_dict["c"]={"hierarchy": 1}
    hierarchy_dict["b"]={"hierarchy": 1}
    hierarchy_dict["d"]={"hierarchy": 0}
    hierarchy_dict["e"]={"hierarchy": 1}
    hierarchy_dict["f"]={"hierarchy": 0}
    nx.set_node_attributes(G, hierarchy_dict)

    count=0
    hierarchy_dict={}
    for node in G_undir.nodes():
        if G_undir.nodes[node]['club']=="Officer" and count <= 10:
            hierarchy_dict[node]={"hierarchy": 0}
            count += 1
        else:
            hierarchy_dict[node]={"hierarchy": 1}


    count=0
    hierarchy_dict={}
    for node in G_undir.nodes():
        if node in [0,33]:
            hierarchy_dict[node]={"hierarchy": 0}
            count += 1
        else:
            hierarchy_dict[node]={"hierarchy": 1}
    nx.set_node_attributes(G_undir, hierarchy_dict)

    G2=nx.florentine_families_graph()
    count=0
    hierarchy_dict={}
    for node in G2.nodes():
        if node in ['Medici','Guadagni','Peruzzi', 'Strozzi' ]:
            hierarchy_dict[node]={"hierarchy": 0}
            count += 1
        else:
            hierarchy_dict[node]={"hierarchy": 1}
    nx.set_node_attributes(G2, hierarchy_dict)

    G3=nx.les_miserables_graph()
    count=0
    hierarchy_dict={}
    for node in G3.nodes():
        if node in ['Valjean','Cosette','Javert', 'Gavroche', 'Fantine', 'Marius' ]:
            hierarchy_dict[node]={"hierarchy": 0}
            count += 1
        else:
            hierarchy_dict[node]={"hierarchy": 1}
    nx.set_node_attributes(G3, hierarchy_dict)

    return G, G_undir,G2,G3


hierarchy_attn_matrix=torch.tensor([[1,0.2],[1,1]])

G, G_undir,G2,G3 = create_test_data()
G = G_undir
#G=G2
losses = []
se_losses = []
pr_losses = []
total_losses = []
lr_list = []
config_file = check_create_folder("configs/hsdne.json")
config = process_config(config_file)

# Create the Agent and pass all the configuration to it then run it..
agent_class = globals()[config.agent]
agent = agent_class(config, G, hierarchy_attention_matrix=hierarchy_attn_matrix)

agent.train()

import pandas as pd
import matplotlib.pyplot as plt
#import scipy as sp
##lr_list = pd.DataFrame(np.array(agent.lr_list)[1:])
#lr_list.plot(title="Learning Rate")
#plt.show()
#for l in agent.losses_dict.keys():
#    losses = pd.DataFrame(np.array(agent.losses_dict[l])[1:])
#    losses.plot(title=l)
#    plt.show()

agent.draw_losses()

agent.measure()
print(agent.measures)
print(agent.measures["rec_map"])
print(agent.measures["emb_map"])
print(agent.measures["rec_l2"])


node_color_dict={}
for idx in list(G.nodes):
    try:
        club = G.nodes[idx]['club']
        if club == "Officer":
            node_color_dict[idx] = "red"
        else:
            node_color_dict[idx] = "blue"
    except:
        node_color_dict[idx]="blue"




agent.draw_embedding(node_color_dict=node_color_dict, xlim=1.2, ylim=1.2)


print(agent.normalize_and_embed())

agent.draw_embedding(node_color_dict=node_color_dict, xlim=1.2, ylim=1.2)
