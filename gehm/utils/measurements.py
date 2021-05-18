from typing import Union, Tuple

import torch
import numpy as np
import networkx as nx

from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm


def aggregate_measures(positions:Union[torch.Tensor, np.ndarray], est_similarities:Union[torch.Tensor, np.ndarray], similarities:Union[torch.Tensor, np.ndarray], cut:float=0.0):

    if isinstance(positions, torch.Tensor):
        positions=positions.numpy()

    if isinstance(est_similarities, torch.Tensor):
        est_similarities=est_similarities.numpy()

    if isinstance(similarities, torch.Tensor):
        similarities=similarities.numpy()

    # Cutoff
    est_similarities[est_similarities<=cut]=0

    measure_dict={}

    ##### Reconstruction Space

    # Precision calculations

    measure_dict["rec_2precision"]=avg_k_precision(similarity=similarities, est_similarity=est_similarities, k=2)
    measure_dict["rec_5precision"]=avg_k_precision(similarity=similarities, est_similarity=est_similarities, k=5)
    measure_dict["rec_map"]=mean_AP(similarity=similarities, est_similarity=est_similarities)

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

    measure_dict["rec_pagerank_overlap"]=sum(pr_sort==pr_sort_est)
    measure_dict["rec_pagerank_l2"]=torch.norm(torch.as_tensor(pr_vals)-torch.as_tensor(pr_est_vals)).numpy() # Use torch here for consistency with the loss measures

    # Reconstruction
    measure_dict["rec_l2"]=float(torch.norm(torch.as_tensor(est_similarities)-torch.as_tensor(similarities)).numpy())

    ##### Embedding Space

    # Distances in embedding space
    pos_distance=torch.cdist(torch.as_tensor(positions),torch.as_tensor(positions))
    pos_distance=row_norm(pos_distance).type(torch.DoubleTensor)


    measure_dict["emb_l2"]=float(torch.norm(torch.as_tensor(pos_distance)-torch.as_tensor(similarities)).numpy())

    measure_dict["emb_2precision"]=avg_k_precision(similarity=similarities, est_similarity=pos_distance, k=2)
    measure_dict["emb_5precision"]=avg_k_precision(similarity=similarities, est_similarity=pos_distance, k=5)
    measure_dict["emb_map"]=mean_AP(similarity=similarities, est_similarity=pos_distance)

    # SE Distance
    se_distance=torch.cdist(torch.as_tensor(similarities),torch.as_tensor(similarities))
    se_distance=row_norm(se_distance).type(torch.DoubleTensor)

    measure_dict["emb_mean_se_cosine"]=torch.nn.functional.cosine_similarity(pos_distance,se_distance, dim=-1).mean().numpy()
    measure_dict["emb_mean_se_l2"] = torch.trace(torch.cdist(se_distance,pos_distance)).mean().numpy()




    return measure_dict


def k_precision_vector(similarity:Union[torch.Tensor, np.ndarray], est_similarity:Union[torch.Tensor, np.ndarray],k:int=5)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Estimates the average of ranked neighborhoods between two adjacency matrices up to a degree
    of k most prominent neighbors.

    Parameters
    ----------
    similarity: Union[torch.Tensor, np.ndarray]
        Original adjacency matrix KxN
    est_similarity: Union[torch.Tensor, np.ndarray]
        Estimated adjacecny matrix KxN
    k: int
        How m

    Returns
    -------
    k_precision_vec, nr_ties, nr_mutual_ties: Tuple(np.ndarray,np.ndarray,np.ndarray)
        Precisions per node,
        number of ties in the ground truth adjacency matrix per node
        number of shared ties in the top k neighborhoods
    """
    def intersect1d_padded(x):
        x, y = np.split(x, 2)
        padded_intersection = -1 * np.ones(x.shape, dtype=int)
        intersection = np.intersect1d(x, y)
        padded_intersection[:intersection.shape[0]] = intersection
        return padded_intersection

    def rowwise_intersection(a, b):
        return np.apply_along_axis(intersect1d_padded,
                                   1, np.concatenate((a, b), axis=1))

    if isinstance(similarity, torch.Tensor):
        similarity=similarity.numpy()

    if isinstance(est_similarity, torch.Tensor):
        est_similarity=est_similarity.numpy()

    est_sort=np.argsort(-est_similarity, axis=1)[:,0:k]
    sim_sort=np.argsort(-similarity, axis=1)[:,0:k]

    nr_ties=(similarity>0).sum(axis=1)
    nr_ties[nr_ties>k]=k

    intersec=rowwise_intersection(est_sort, sim_sort)
    nr_mutual_ties=(intersec>-1).sum(axis=1)

    nr_mutual_ties=np.min([nr_ties,nr_mutual_ties], axis=0)

    k_precision_vec=nr_mutual_ties/k

    return k_precision_vec, nr_ties, nr_mutual_ties

def avg_k_precision(similarity:Union[torch.Tensor, np.ndarray], est_similarity:Union[torch.Tensor, np.ndarray],k:int=5):

    return k_precision_vector(similarity,est_similarity,k)[0].mean()

def mean_AP(similarity:Union[torch.Tensor, np.ndarray], est_similarity:Union[torch.Tensor, np.ndarray]):

    ap_list=[]
    k_list=[]
    for k in range(1,similarity.shape[0]):
        k_precision_vec, nr_ties, nr_mutual_ties=k_precision_vector(similarity,est_similarity,k)
        has_mutual_ties=np.array(nr_mutual_ties>0,dtype=int).sum() # When averaging, we need to discard instances where there are no mutual ties
        if has_mutual_ties>0:
            map_k=k_precision_vec.sum()/has_mutual_ties
            k_list.append(k)
        else:
            map_k=0
        ap_list.append(map_k)

    map=np.array(ap_list).sum()/len(k_list)

    return map