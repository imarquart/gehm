import os

from gehm.model.positions import Disk2

#os.chdir("../")

from gehm.datasets.nx_datasets import *
from gehm.utils.config import process_config
from gehm.utils.file_helpers import check_create_folder
from gehm.agents.sdne import SDNEAgent

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
config_file = check_create_folder("configs/sdne.json")
config = process_config(config_file)

# Create the Agent and pass all the configuration to it then run it..
agent_class = globals()[config.agent]
agent = agent_class(config, G)

agent.train()

import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
lr_list = pd.DataFrame(np.array(agent.lr_list)[1:])
lr_list.plot(title="Learning Rate")
plt.show()
for l in agent.losses_dict.keys():
    losses = pd.DataFrame(np.array(agent.losses_dict[l])[1:])
    losses.plot(title=l)
    plt.show()

agent.measure()
print(agent.measures)
print(agent.measures["rec_map"])
print(agent.measures["emb_map"])
print(agent.measures["rec_l2"])



predictions,losses = agent.predict()
nodes,positions,similarities=predictions
similarities_cut=similarities
similarities_cut[similarities_cut<=0.1]=0
G_est=nx.from_numpy_array(similarities_cut)
G_norm=nx.from_numpy_array(agent.dataset.sim1.numpy())

aa=((torch.tensor(similarities)-torch.min(torch.tensor(similarities),dim=0, keepdims=True)[0])/(torch.max(torch.tensor(similarities),dim=0, keepdims=True)[0]-torch.min(torch.tensor(similarities),dim=0, keepdims=True)[0])).numpy()

bb=agent.dataset.sim1.numpy()

pr=pd.Series(nx.pagerank(G))
pr_est=pd.Series(nx.pagerank(G_est))
s1=np.argsort(pr)
s2=np.argsort(pr_est)
pageranks=pd.DataFrame([pr,pr_est]).T
pageranks=pd.DataFrame([s1,s2]).T
diff=sp.spatial.distance.norm(similarities-agent.dataset.sim1.numpy())
sp.spatial.distance.cdist(similarities,agent.dataset.sim1.numpy()).mean()

asdf=pd.DataFrame(positions, index=nodes, columns=["x","y"])

asdf=(asdf - np.mean(asdf)) / np.std(asdf)
d2=Disk2()
d2pos=d2(torch.tensor(asdf.to_numpy())).numpy()
asdf=pd.DataFrame(d2pos, index=nodes, columns=["x","y"])
figure, axes = plt.subplots()
Drawing_colored_circle = plt.Circle((0, 0), 1, fill=False)
plt.scatter(asdf.x, asdf.y)
axes.set_aspect(1)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
asdf["drawn"]=0
for ind in asdf.index:
    row = asdf.loc[ind, :]
    idx = agent.dataset.node_idx_dict[ind]
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
        #print("{}: {} - {},{}".format(ind,neighbors,xp,yp))
        plt.text(
            x=row.x + xp,
            y=row.y + yp,
            s=idx,
            fontdict=dict(color="black", size=10),
            bbox=dict(facecolor="blue", alpha=0.1),
        )
    asdf.loc[ind, "drawn"]=1

axes.add_artist(Drawing_colored_circle)
plt.title("Embedding")
plt.show()

