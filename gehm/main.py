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
from gehm.model.positions import Position, Circle, Disk, Disk2
from gehm.losses.loss_functions import *
from gehm.datasets.nx_datasets import *
from gehm.model.gehm import OneLevelGehm,OneLevelGehm2
from tqdm import tqdm


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

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
G=G_undir
losses=[]
firstlosses=[]
secondlosses=[]
decoderlosses=[]
lr_list=[]

nr_nodes = len(G.nodes)
nr_epochs = 50000
batch_size = nr_nodes
dataset = nx_dataset_onelevel(G)
loss_fn = WeightedLoss(embedding_weight=0.8,first_deg_weight=0.5, beta=1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
gehm = OneLevelGehm2(batch_size, nr_nodes, Disk2, torch.nn.Tanh,nr_transformers=6,nr_heads=batch_size)

#optimizer=torch.optim.SGD(gehm.parameters(), lr=0.0001, momentum=0.99)
#optimizer = torch.optim.Adam(gehm.parameters(),lr=0.0001,weight_decay=0.00001,amsgrad=True)
optimizer = torch.optim.Adam(gehm.parameters(),lr=0.0001,amsgrad=True)
#scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.0001, max_lr=0.0005,step_size_up=100,mode="exp_range",gamma=0.99)
desc=""
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
step_size=50, gamma=0.99)
epoch_loss=torch.tensor(0)
for epoch in range(0, nr_epochs):
    gehm.train()

    pbar = tqdm(enumerate(dataloader),desc=desc, position=0, leave=False)
    start_time = time.time()
    if epoch_loss > 0:
        losses.append(epoch_loss.detach().numpy())
        firstlosses.append(firstloss.detach().numpy())
        secondlosses.append(secondloss.detach().numpy())
        decoderlosses.append(decoderloss.detach().numpy())
    epoch_loss=0
    firstloss=0
    secondloss=0
    decoderloss=0
    
    for i, data in pbar:
        prepare_time = start_time - time.time()
        optimizer.zero_grad()

        node_ids, sim1, sim2 = data
        # This needs collate fn
        #index = torch.sort(node_ids)[0].long()
        #sim1 = sim1[:, index]
        #sim2 = sim2[:, index]
        if torch.norm(sim1) > 0:
            positions,est_sim = gehm(sim1)
            kl_loss, firstloss,secondloss,decoderloss = loss_fn(similarity=sim1,similarity2=sim2, positions=positions, est_similarity=est_sim)
            kl_loss.backward()
            optimizer.step()
            epoch_loss=epoch_loss+kl_loss
            
            desc="KL Weighted loss: {}, LR:{},  epoch: {}/{}:".format(epoch_loss, scheduler.get_last_lr(),epoch, nr_epochs)
            

        start_time = time.time()
    lr_list.append(scheduler.get_last_lr()[0] if isinstance(scheduler.get_last_lr(),list) else scheduler.get_last_lr().detach().numpy())
    scheduler.step()


import pandas as pd
import matplotlib.pyplot as plt
lr_list=pd.DataFrame(np.array(lr_list)[1:])
lr_list.plot()
losses=pd.DataFrame(np.array(losses)[1:])
losses.plot()
firstlosses=pd.DataFrame(np.array(firstlosses)[1:])
secondlosses=pd.DataFrame(np.array(secondlosses)[1:])
decoderlosses=pd.DataFrame(np.array(decoderlosses)[1:])
firstlosses.plot()
secondlosses.plot()
decoderlosses.plot()
arr=[]
with torch.no_grad():
    gehm.eval()
    kl_loss=0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, data in enumerate(dataloader):
        node_ids, sim1, sim2 = data
        positions,est_sim = gehm(sim1)
        kl_loss, firstloss,secondloss,decoderloss = loss_fn(similarity=sim1,similarity2=sim2, positions=positions, est_similarity=est_sim)
        asdf = pd.DataFrame(positions.detach().numpy())
        asdf.index=np.array(list(G.nodes))
        asdf.columns = ["x", "y"]
        asdf["loss"]=kl_loss.detach().numpy()
        arr.append(asdf)

asdf=pd.concat(arr)


figure, axes = plt.subplots()
Drawing_colored_circle = plt.Circle((0, 0), 1, fill=False)
plt.scatter(asdf.x, asdf.y)
axes.set_aspect(1)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
asdf["drawn"]=0
for ind in asdf.index:
    row = asdf.loc[ind, :]
    x=row.x
    y=row.y
    close_x=np.where(np.isclose(asdf.x,x,atol=0.1))[0]
    close_y=np.where(np.isclose(asdf.y,y,atol=0.1))[0]
    closeset=np.intersect1d(close_x,close_y)
    neighbors=np.sum(asdf.iloc[closeset,:].drawn)+1
    if neighbors <= 4:
        npd=np.array([1,1,2,2])
        nn=neighbors%4
        xp=np.power(-1,npd[nn-1])*(max(1,neighbors-4))*0.1
        yp=np.power(-1,nn)*(max(1,neighbors-4))*0.1
        print("{}: {} - {},{}".format(ind,neighbors,xp,yp))
        plt.text(
            x=row.x + xp,
            y=row.y + yp,
            s=ind,
            fontdict=dict(color="black", size=10),
            bbox=dict(facecolor="blue", alpha=0.1),
        )
    asdf.loc[ind, "drawn"]=1

axes.add_artist(Drawing_colored_circle)
plt.title("Embedding")
plt.show()
