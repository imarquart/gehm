import os

from gehm.model.positions import Disk2

#os.chdir("../")
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from gehm.datasets.nx_datasets import *
from gehm.utils.config import process_config
from gehm.utils.file_helpers import check_create_folder
from gehm.agents.sdne import SDNEAgent,tSDNEAgent
from gehm.utils.measurements import aggregate_measures


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

sdne_df=[]
tsdne_df=[]

epoch_list=[10,100,1000,2000,5000]
#epoch_list=list(np.logspace(0.1, 4.2, num=25, dtype=int))


#epoch_list=list(np.logspace(0.1, 2, num=10, dtype=int))
for nr_epochs in epoch_list:
    epoch_dict={}
    epoch_dict["model"]="tSDNE"
    epoch_dict["epoch"]=nr_epochs

    config_file = check_create_folder("configs/tsdne.json")
    config = process_config(config_file)

    config.nr_epochs=nr_epochs

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config, G)

    agent.train()
    agent.measure()


    epoch_dict["rec_map"]=agent.measures["rec_map"]
    epoch_dict["emb_map"]=agent.measures["emb_map"]
    epoch_dict["rec_l2"]=agent.measures["rec_l2"]
    epoch_dict["emb_l2"]=agent.measures["emb_l2"]
    epoch_dict["rec_2precision"]=agent.measures["rec_2precision"]
    epoch_dict["emb_2precision"]=agent.measures["emb_2precision"]
    epoch_dict["rec_5precision"]=agent.measures["rec_5precision"]
    epoch_dict["emb_5precision"]=agent.measures["emb_5precision"]
    tsdne_df.append(epoch_dict)

    agent.finalize()
    del agent
    del config

for nr_epochs in epoch_list:
    epoch_dict={}
    epoch_dict["model"]="SDNE"
    epoch_dict["epoch"]=nr_epochs

    config_file = check_create_folder("configs/sdne.json")
    config = process_config(config_file)

    config.nr_epochs=nr_epochs

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config, G)

    agent.train()
    agent.measure()


    epoch_dict["rec_map"]=agent.measures["rec_map"]
    epoch_dict["emb_map"]=agent.measures["emb_map"]
    epoch_dict["rec_l2"]=agent.measures["rec_l2"]
    epoch_dict["emb_l2"]=agent.measures["emb_l2"]
    epoch_dict["rec_2precision"]=agent.measures["rec_2precision"]
    epoch_dict["emb_2precision"]=agent.measures["emb_2precision"]
    epoch_dict["rec_5precision"]=agent.measures["rec_5precision"]
    epoch_dict["emb_5precision"]=agent.measures["emb_5precision"]
    sdne_df.append(epoch_dict)
    agent.finalize()
    del agent
    del config


results_tsdne=pd.DataFrame(tsdne_df)
results_sdne=pd.DataFrame(sdne_df)

import matplotlib.pyplot as plt

cols=np.array(list(results_tsdne.columns))
cols=np.setdiff1d(cols, np.array(["measure","epoch","model"]))

for col in cols:
    fig, axs = plt.subplots(1,1)
    axs.plot(results_tsdne.epoch, results_tsdne[col], label = "tSDNE", color='red')
    plt.plot(results_sdne.epoch, results_sdne[col], label = "SDNE", color='blue')
    axs.legend()
    axs.set_title(col)
    axs.set_xlabel('epoch')
    plt.show()