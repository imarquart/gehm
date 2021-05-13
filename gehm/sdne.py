import os
import time
os.chdir("../")

import networkx as nx
import numpy as np
import pytest

from numpy import ndarray, cos, sin
from torch import tensor
from typing import Union, Optional
import torch
import numpy as np
from torch import int16
from torch.utils.data import DataLoader
import networkx as nx
from gehm.losses.sdne_loss_functions import *
from gehm.datasets.nx_datasets import *
from gehm.model.sdne import SDNEmodel
from gehm.agents.sdne import SDNEAgent
from gehm.utils.config import process_config
from gehm.utils.file_helpers import check_create_folder
from tqdm import tqdm




def create_test_data():
    G_undir = nx.karate_club_graph()
    G = nx.Graph()
    G.add_edge("a", "b", weight=0.6)
    G.add_edge("a", "c", weight=0.2)
    G.add_edge("c", "d", weight=0.1)
    G.add_edge("c", "e", weight=0.7)
    G.add_edge("c", "f", weight=0.9)
    G.add_edge("a", "d", weight=0.3)

    return G, G_undir


G, G_undir = create_test_data()
G = G_undir
losses = []
se_losses = []
pr_losses = []
total_losses = []
lr_list = []
config_file=check_create_folder("configs/sdne.json")
config = process_config(config_file)

# Create the Agent and pass all the configuration to it then run it..
agent_class = globals()[config.agent]
agent = agent_class(config, G)



agent.train()

import pandas as pd
import matplotlib.pyplot as plt

lr_list = pd.DataFrame(np.array(lr_list)[1:])
lr_list.plot(title="Learning Rate")
losses = pd.DataFrame(np.array(losses)[1:])
losses.plot(title="Total Loss")
pr_losses = pd.DataFrame(np.array(pr_losses)[1:])
se_losses = pd.DataFrame(np.array(se_losses)[1:])

pr_losses.plot(title="Proximity Embedding Loss")
se_losses.plot(title="SE Prediction Loss")




arr = []
with torch.no_grad():
    sdne.eval()
    sdne = sdne.to("cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, data in enumerate(dataloader):
        node_ids, sim1, sim2 = data
        node_ids = node_ids.to("cpu")
        sim1 = sim1.to("cpu")
        positions, est_sim = sdne(sim1)
        asdf = pd.DataFrame(positions.detach().numpy())
        indecies=[dataset.node_idx_dict[int(x)] for x in node_ids]
        asdf.index = indecies
        arr.append(asdf)



