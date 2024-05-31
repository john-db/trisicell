import copy
from decimal import Decimal

import numpy as np
import numpy.linalg as la
from scipy.special import softmax
from sklearn.metrics.pairwise import pairwise_distances


def draw_sample_clt(P, greedy, names_to_cells, clade, c=1, coef=2):
    r"""
    Draw sample clt.

    :param P:
    :param greedy:
    :param c: gaussian kernel parameter
    :param coef:
    :return: edges, subtrees, prior_prob
    prior_prob in the latex: Prob_{T\sim E}[T]
    """

    edges, prior_prob = clt_sample_rec(P, greedy, c, coef=coef, names_to_cells=names_to_cells, clade=clade)
    n_cells = P.shape[0]
    n_nodes = 2 * n_cells - 1
    edges_map = {a: (b, d) for a, b, d in edges}
    subtrees = []
    for i in range(n_nodes):
        if i not in edges_map:  # leaf
            row = np.zeros(n_cells, dtype=np.int8)
            row[i] = 1
        else:
            pair = edges_map[i]
            row = subtrees[pair[0]] + subtrees[pair[1]]  # logical_or
        subtrees.append(row)
    return edges, subtrees, prior_prob


def clt_sample_rec(
    P,
    greedy,
    c,
    names_to_cells,
    clade,
    names=None,
    namecount=None,
    coef=2,
    prior_prob=None,
    join_prob_matrix=None,
):
    """
    Clt sample recursion.

    O(n^2 m^2)

    n: depth of the recursion two edges at each call
    nm^2 : runtime of each call

    can be improved to
    :param P: probability matrix
    :param greedy: sample or max
    :param c: gaussian kernel parameter
    :param names: for rec
    :param namecount: for rec
    :param coef: coef between dist and common mut
    :param prior_prob: for rec
    :return: edges, prior_prob
    """
    if len(P) > 1:
        print("\n\t------- Next iteration --------\n")
    # TODO make this faster by not recalculating
    if prior_prob is None:
        prior_prob = Decimal(1.0)
    if P.shape[0] == 1:
        return [], prior_prob
    if names is None:
        names = list(range(P.shape[0]))
        namecount = P.shape[0]

    if join_prob_matrix is None:

        def join_neg_priority(a, b):  # smaller values should be joined first
            return la.norm(a - b) - row_leafness_score(a, b) * coef

        dist = pairwise_distances(P, metric=join_neg_priority)  # O(n m^2)
        dist = dist.astype(np.float64)
        np.fill_diagonal(dist, np.inf)

        # This block adjusts c if dist/2c is too big for sotfmax.
        c_rec = c
        for _ in range(10):
            sim = softmax(-dist / (2 * c_rec))
            if not np.any(np.isnan(sim)):
                break
            c_rec *= 2.0
    else:
        # add new row to join_prob_matrix to get sim
        pass
    prob = sim
    if greedy:
        pair = np.unravel_index(np.argmax(prob), prob.shape)
    else:
        flat_probs = np.float64(prob.flat)
        ind = np.random.choice(len(flat_probs), p=flat_probs)
        pair = np.unravel_index(ind, prob.shape)
    
    # for i in range(len(prob.flat)):
    #     pair_i = np.unravel_index(i, prob.shape)
    #     cells1 = names_to_cells[int(names[pair_i[0]])]
    #     cells2 = names_to_cells[int(names[pair_i[1]])]
    #     print("\t" + cells1 + " AND " + cells2 + ": " + str(prob.flat[i]))

    index_and_ps = []
    for i in range(len(prob.flat)):
        pair_i = np.unravel_index(i, prob.shape)
        cells1 = names_to_cells[int(names[np.max(pair_i)])]
        cells2 = names_to_cells[int(names[np.min(pair_i)])]

        cells1ls = cells1.split(",")
        cells2ls = cells2.split(",")

        int1 = set(cells1ls) & set(clade)
        int2 = set(cells2ls) & set(clade)

        nbg = None
        if len(int1) == 0 and len(int2) == 0:
            nbg = "NEUTRAL"
        elif set(clade).issubset(int1) or set(clade).issubset(int2):
            nbg = "NEUTRAL"
        elif len(int1) == len(cells1ls) and len(int2) == len(cells2ls):
            nbg = "GOOD"
        else:
            nbg = "BAD"

        index_and_ps.append((prob.flat[i], cells1, cells2, nbg))
    index_and_ps = sorted(list(set(index_and_ps)), reverse=True)
    strings = []
    for i in range(len(index_and_ps)):
        temp = ""
        tup = index_and_ps[i]
        if tup[1] == tup[2]: 
            pass
        else:
            temp += "\t" + tup[1] + " and " + tup[2] + ": " + str("{:.5E}".format(tup[0])) + " " + tup[3]
            strings.append(temp)
    maxlen = len(max(strings, key=len))
    strings = [s.ljust(maxlen) for s in strings]

    # for i in range(len(strings) // 2):
    #     temp = ""
    #     for j in range(2):
    #         if i + j < len(strings) - 1:
    #             temp += strings[i+j] + " \t"
    #     print(temp)

    cells1 = names_to_cells[int(names[np.max(pair)])]
    cells2 = names_to_cells[int(names[np.min(pair)])]
    cells1ls = cells1.split(",")
    cells2ls = cells2.split(",")
    int1 = set(cells1ls) & set(clade)
    int2 = set(cells2ls) & set(clade)
    nbg = None
    if len(int1) == 0 and len(int2) == 0:
            nbg = "NEUTRAL"
    elif set(clade).issubset(int1) or set(clade).issubset(int2):
        nbg = "NEUTRAL"
    elif len(int1) == len(cells1ls) and len(int2) == len(cells2ls):
        nbg = "GOOD"
    else:
        nbg = "BAD"

    perline = 1
    for i in range(len(strings) // perline):
        temp = ""
        for j in range(perline):
            if i + j < len(strings) - 1:
                temp += strings[i+j] + " \t"
        print(temp)

    print("\nSum of distribution = " + str(sum(prob.flat)))
    
    
    print("\n\tPair chosen: " + cells1 + " and " + cells2 + ": " + str("{:.5E}".format(prob.flat[ind])) + " " + nbg)


    

    # conversion from numpy.float128 to Decimal is not supported
    prior_prob = prior_prob * Decimal(np.float64(prob[pair]))
    P_new = np.delete(P, pair, axis=0)  # remove two rows
    P_new = np.append(
        P_new, np.minimum(P[pair[0]], P[pair[1]]).reshape(1, -1), axis=0
    )  # add one row that has only common ones
    new_edge = [namecount, names[pair[0]], names[pair[1]]]

    new_names = copy.copy(names)
    del new_names[np.max(pair)]
    del new_names[np.min(pair)]
    new_names.append(namecount)
    newnamecount = namecount + 1

    parent_cell = names_to_cells[int(names[np.max(pair)])] + "," + names_to_cells[int(names[np.min(pair)])]
    new_names_to_cells = names_to_cells.copy() + [parent_cell]

    edges, prior_prob = clt_sample_rec(
        P_new, greedy, c, new_names_to_cells, clade, new_names, newnamecount, coef, prior_prob
    )
    edges.append(new_edge)
    return edges, prior_prob


def row_leafness_score(row_a, row_b):
    return np.sum(np.minimum(row_a, row_b))
