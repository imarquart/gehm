from typing import Union

import torch
import numpy as np
import networkx as nx

from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm


def aggregate_measures(positions:Union[torch.Tensor, np.ndarray], est_similarities:Union[torch.Tensor, np.ndarray], similarities:Union[torch.Tensor, np.ndarray], cut:float=0.1):

    if isinstance(positions, torch.Tensor):
        positions=positions.numpy()

    if isinstance(est_similarities, torch.Tensor):
        est_similarities=est_similarities.numpy()

    if isinstance(similarities, torch.Tensor):
        similarities=similarities.numpy()

    # Cutoff
    est_similarities[est_similarities<=cut]=0

    # PageRanks
    G_est=nx.from_numpy_array(est_similarities)
    G_norm=nx.from_numpy_array(similarities)
    pr=nx.pagerank(G_norm)
    pr_index=np.array(list(pr.keys()))
    pr_vals=np.array(list(pr.values()))
    pr_vals=pr_vals[np.argsort(pr_index)] # Sort by node index

    pr_est=nx.pagerank(G_est)
    pr_est_index=np.array(list(pr_est.keys()))
    pr_est_vals=np.array(list(pr_est.values()))
    pr_est_vals=pr_est_vals[np.argsort(pr_est_index)] # Sort by node index

    # Now sort by pagerank
    pr_sort=np.argsort(pr_vals)
    pr_sort_est=np.argsort(pr_est_vals)

    pagerank_overlap=sum(pr_sort==pr_sort_est)
    pagerank_l2=torch.norm(torch.tensor(pr_vals)-torch.tensor(pr_est_vals)).numpy() # Use torch here for consistency with the loss measures

    # Reconstruction
    reconstruction_l2=torch.norm(torch.tensor(est_similarities)-torch.tensor(similarities)).numpy()

    # Distances in embedding space
    pos_distance=torch.cdist(torch.tensor(positions),torch.tensor(positions))
    pos_distance=row_norm(pos_distance).type(torch.DoubleTensor)

    # SE Distance
    se_distance=torch.cdist(torch.tensor(similarities),torch.tensor(similarities))
    se_distance=row_norm(se_distance).type(torch.DoubleTensor)

    mean_se_cosine=torch.nn.functional.cosine_similarity(pos_distance,se_distance, dim=-1).mean().numpy()
    mean_se_l2 = torch.trace(torch.cdist(se_distance,pos_distance)).mean().numpy()


    return pagerank_overlap, pagerank_l2, reconstruction_l2, mean_se_cosine, mean_se_l2