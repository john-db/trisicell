from decimal import Decimal

import numpy as np
from tqdm import tqdm

from trisicell.tl.partition_function._clt_sampler import draw_sample_clt


def cell_lineage_tree_prob(P, subtrees):
    r"""
    Calculate Prob_{A\sim P}[A\in T] in O(n m^2).

    :param P:
    :param subtrees: cell lineage tree
    :return: Probability of this tree in the original distribution( based on P)
    """

    return_value = Decimal(1)
    for j in range(P.shape[1]):
        temp = Decimal(0)
        col = P[:, j]
        for v in subtrees:
            temp += Decimal(np.product(col * v + (1 - col) * (1 - v)))
        return_value *= temp
    return return_value


def pf_cond_on_one_tree(P, subtrees, cond_c, cond_m):
    r"""
    Prob_{A\sim P}[\subtree(c, R, A)\cap A\in G| A\in T] in O(n^2).

    :param P:
    :param subtrees: cell lineage tree  n x (2n+1)
    :param cond_c: set of cells
    :param cond_m: one mutation
    :return: conditioned on the given tree what is probability of the given partition
        based on P numerator, denominator are returned separately here
    """

    denominator = Decimal(0)
    numerator = Decimal(0)
    col = P[:, cond_m]
    for v in subtrees:
        prob = Decimal(np.product(col * v + (1 - col) * (1 - v)))
        denominator += prob
        if np.array_equal(v, cond_c):
            numerator = prob
    return numerator, denominator


def get_samples(P, n_samples, names_to_cells, cells, eps, delta, divide, coef, disable_tqdm=True):
    r"""
    N_s *.

    :param P:
    :param n_samples:
    :return: `n_samples` of cell lineage trees.
        The function returns three lists with this length:
        edges_list: for each sample: (n-1) wedges. Each wedge
            [parent, left_child, right_child]
        subtrees_list: for each sample: 2n+1 number of cell subset
            masks, i.e., potential column.
        tree_our_prob_list: for each sample: the probability of us sampling
            the tree. Prob_{T\sim E}[T]
    """

    P = P.astype(np.float64)
    edges_list = []
    subtrees_list = []
    tree_our_prob_list = []

    rng = np.random.default_rng(seed=0)
    for _ in tqdm(
        range(n_samples), ascii=True, ncols=100, desc="Sampling", disable=disable_tqdm
    ):
        # print("Sample " + str(_))
        n_to_c = names_to_cells.copy()

        edges, subtrees, prior_prob = draw_sample_clt(P, False, c=1, eps=eps, delta=delta, divide=divide, coef=coef, names_to_cells=n_to_c, clade=cells, rng=rng)
        edges_list.append(edges)
        subtrees_list.append(subtrees)
        tree_our_prob_list.append(prior_prob)
    
    # clades_list = sorted([["".join([str(i) for i in st]) for st in subtrees] for subtrees in subtrees_list])
    # clades_list = [",".join(clade) for clade in clades_list]
    return edges_list, subtrees_list, tree_our_prob_list


def get_samples_info(
    P, my_cell, my_mut, n_samples, subtrees_list=None, disable_tqdm=True
):
    r"""
    Run some processes on the given samples and returns some raw data.

    If not given first gets the samples.

    If sampling was done internally, it outputs the full sample information.

    If not given_samples:
        O(N * (T_nj * T_obj))
        T_obj = n m^2
        T_nj = n^2 m^2
    Else:
        O(N * n^2)

    :param P: A probability matrix:
            where P_{i,j} is the probability of i\th cell having j\th mutation in
            the latent original matrix.
            In other words, if O is the original matrix and I is the measured
            (observed) matrix then, P_{i,j} = P[O_{i,j}=1|<I_{i,j}, \alpha, \beta>].
    :param my_cell:
    :param my_mut:
    :param n_samples:
    :param subtrees_list: presampled trees. If None new samples will be drawn
    :return:
        if given_samples:   (i.e., subtrees_list is not None)
            pf_cond_list, tree_origin_prob_list
        else:               (i.e., subtrees_list is None)
            pf_cond_list, tree_origin_prob_list, edges_list, subtrees_list,
            tree_our_prob_list
    """

    given_samples = subtrees_list is not None
    if not given_samples:
        edges_list, subtrees_list, tree_our_prob_list = get_samples(P, n_samples)

    pf_cond_list = []
    tree_origin_prob_list = []
    cond_c = np.zeros(P.shape[0], dtype=np.int8)
    cond_c[my_cell] = 1

    for i in tqdm(
        range(n_samples), ascii=True, ncols=100, desc="Sampling", disable=disable_tqdm
    ):
        numerator, denominator = pf_cond_on_one_tree(
            P, subtrees_list[i], cond_c=cond_c, cond_m=my_mut
        )
        pf_cond_list.append(numerator / denominator)

        origin_prob = cell_lineage_tree_prob(P, subtrees_list[i])
        tree_origin_prob_list.append(origin_prob)

    if given_samples:
        return pf_cond_list, tree_origin_prob_list
    else:
        return (
            pf_cond_list,
            tree_origin_prob_list,
            edges_list,
            subtrees_list,
            tree_our_prob_list,
        )


def process_samples(
    pf_cond_list, tree_origin_prob_list, tree_our_prob_list, n_batches=None
):
    """
    Combine the data corresponding to the sampled trees and output the partition.

    Estimate according to the formula in the paper. One can see this as just
    a simple weighted average.

    Features:
        1. Debiasing is implemented through weights
            (in favour of original probability and against probability of our sampling
        2. Normalization: instead of calculating exact denominator given in the paper.
            We divide the value by summation of the weights.

    :param pf_cond_list:
    :param tree_origin_prob_list:
    :param tree_our_prob_list:
    :param n_batches:
    :return: just one probability through weighted average
    """

    n_samples = len(pf_cond_list)
    estimates = []
    n_batches_internal = n_batches if n_batches is not None else 1
    interval_len = n_samples // n_batches_internal
    for batch_ind in range(n_batches_internal):
        numerator = 0
        denominator = 0
        samples_interval_start = batch_ind * interval_len
        samples_interval_end = min((batch_ind + 1) * interval_len, n_samples)
        assert samples_interval_start < samples_interval_end

        ls_corrected = []
        ls_raw = []
        for i in range(samples_interval_start, samples_interval_end):
            # numerator += (
            #     pf_cond_list[i] * tree_origin_prob_list[i] / tree_our_prob_list[i]
            # )
            # denominator += tree_origin_prob_list[i] / tree_our_prob_list[i]
            # ls_corrected += [(pf_cond_list[i] * tree_origin_prob_list[i] / tree_our_prob_list[i], tree_origin_prob_list[i] / tree_our_prob_list[i], pf_cond_list[i])]

            numerator += (
                pf_cond_list[i] * tree_origin_prob_list[i]
            )
            denominator += tree_origin_prob_list[i]
            ls_raw += [(pf_cond_list[i] * tree_origin_prob_list[i], tree_origin_prob_list[i], pf_cond_list[i])]
        
        ls_raw = sorted(ls_raw, reverse = True, key = lambda x: x[1])
        ls_corrected = sorted(ls_corrected, reverse = True, key = lambda x: x[1])
        estimates.append(numerator / denominator)

        temp = 0
        for i in range(1, len(ls_raw)):
            temp += ls_raw[i][0]


        # print("tree probs")
        # for i in range(1, 10):
        #     print(tree_our_prob_list[i])
        # print("numerator: " + str(numerator) + "denominator: " + str(denominator))
        # print(estimates[0])
    assert len(estimates) >= 1
    if n_batches is None:
        return estimates[0]
    else:
        return estimates
