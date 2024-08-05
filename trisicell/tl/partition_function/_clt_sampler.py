import copy
from decimal import Decimal

import numpy as np
import numpy.linalg as la
from scipy.special import softmax
from sklearn.metrics.pairwise import pairwise_distances


def draw_sample_clt(P, greedy, names_to_cells, clade, c=1, eps=.1, delta=0.5, divide=False, coef=2, rng=None):
    r"""
    Draw sample clt.

    :param P:
    :param greedy:
    :param c: gaussian kernel parameter
    :param coef:
    :return: edges, subtrees, prior_prob
    prior_prob in the latex: Prob_{T\sim E}[T]
    """
    edges, prior_prob, choices = clt_sample_rec(P, greedy, c, coef=coef, eps=eps, delta=delta, divide=divide, names_to_cells=names_to_cells, clade=clade, rng=rng)
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
    
    # return edges, subtrees, prior_prob

    # This corrects for there being multiple ways to get the same tree
    subtrees_cell_list = []
    for subtree in subtrees:
        ls = []
        for i in range(len(subtree)):
            if subtree[i] == 1:
                ls += [names_to_cells[i]]
        ls = sorted(ls)
        subtrees_cell_list += [ls]
    subtrees_cell_list = sorted(subtrees_cell_list)
    edges2, norm_factor, choices2 = simulate_clt_sample_rec(P, greedy, c, coef=coef, eps=eps, delta=delta, divide=divide, names_to_cells=names_to_cells, clade=clade, choices_made=choices, subtrees_cell_list=subtrees_cell_list)
    prob = prior_prob / norm_factor
    return edges, subtrees, prob


def simulate_clt_sample_rec(
    P,
    greedy,
    c,
    names_to_cells,
    clade,
    choices_made,
    subtrees_cell_list,
    names=None,
    namecount=None,
    eps=.1,
    delta=0.5,
    divide=False,
    coef=2,
    first=True,
    norm_factor=None,
    join_prob_matrix=None
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
    log = ""

    if len(P) > 1 and not first:
        log += "\t------- Next iteration --------" + "\n"
    # TODO make this faster by not recalculating
    if norm_factor is None:
        norm_factor = Decimal(1.0)
    if P.shape[0] == 1:
        return [], norm_factor, []
    if names is None:
        names = list(range(P.shape[0]))
        namecount = P.shape[0]

    if join_prob_matrix is None:

        def join_neg_priority(a, b):  # smaller values should be joined first
            return la.norm(a - b) - row_leafness_score(a, b) * coef
        
            # norm_a = la.norm(a)
            # norm_b = la.norm(b)
            # norm_max = max(norm_a, norm_b)
            # return (1 - coef) * la.norm((a / norm_max) - (b / norm_max)) - coef * cosine(a, b)
            # return la.norm(a - b) - np.dot(a,b) * coef

        dist = pairwise_distances(P, metric=join_neg_priority)  # O(n m^2)
        dist = dist.astype(np.float64)
        np.fill_diagonal(dist, np.inf)
        save = dist.copy()

        if divide:
            #divide to normalize to max:0 min: -eps
            dist = -dist
            dist = dist - max(dist.flat)
            dabs_max = 0
            for entry in dist.flat:
                if entry != float('-inf') and entry != float('inf') and abs(entry) > dabs_max:
                    dabs_max = abs(entry)

            if dabs_max != 0:
                dist = -dist * (eps / dabs_max)
            else:
                dist = -dist
        else:
            #change the base of the softmax exponentation
            dist = -dist
            dist = dist - max(dist.flat)
            dist = -dist * np.log(1 + eps)

        

        

        # This block adjusts c if dist/2c is too big for sotfmax.
        c_rec = c
        # John: I don't think this is necessary
        # for _ in range(10):
        #     sim = softmax(-dist / (2 * c_rec))
        #     if not np.any(np.isnan(sim)):
        #         break
        #     c_rec *= 2.0

        # John: cut off values below x%
        sim = softmax(-dist)
        # distcopy = dist.copy()
        for i in range(len(sim.flat)):
            # if sim.flat[i] < 0.02:
            #     distcopy.flat[i] = float('inf')
            if sim.flat[i] < 0.000002:
                sim.flat[i] = 0
        if np.dot(sim.flat, sim.flat) == 0:
            sim = softmax(-dist)
        else:
            sim = sim / sum(sim.flat)
        # sim = softmax(-distcopy)
    else:
        # add new row to join_prob_matrix to get sim
        pass
    prob = sim

    p_ab = None
    if greedy:
        pair = np.unravel_index(np.argmax(prob), prob.shape)
    else:
        flat_probs = np.float64(prob.flat)

        # np.random.seed(seed=None)

        # ind = np.random.choice(len(flat_probs), p=flat_probs)
        ind = choices_made[0]
        choices_made = choices_made[1:]
        pair = np.unravel_index(ind, prob.shape)

        p_ab = flat_probs[ind]

    accum = 0
    for i in range(len(prob.flat)):
        pair_i = np.unravel_index(i, prob.shape)
        cells1 = names_to_cells[int(names[np.max(pair_i)])]
        cells2 = names_to_cells[int(names[np.min(pair_i)])]

        cells_ls = cells1.split(",") + cells2.split(",")
        cells_ls = sorted(cells_ls)

        if cells_ls in subtrees_cell_list:
            accum += np.float64(prob.flat)[i]

    q_i = p_ab / accum
    
    # conversion from numpy.float128 to Decimal is not supported
    norm_factor = norm_factor * Decimal(np.float64(q_i))
    P_new = np.delete(P, pair, axis=0)  # remove two rows
    # P_new = np.append(
    #     P_new, np.minimum(P[pair[0]], P[pair[1]]).reshape(1, -1), axis=0
    # )  # add one row that has only common ones
    # P_new = np.append(
    #     P_new, np.minimum(P[pair[0]], P[pair[1]]).reshape(1, -1), axis=0
    # )  # add one row that has only common ones

    new_row = delta * np.minimum(P[pair[0]], P[pair[1]]) + (1 - delta) * np.maximum(P[pair[0]], P[pair[1]])
    P_new = np.append(
        P_new, new_row.reshape(1, -1), axis=0
    )
    new_edge = [namecount, names[pair[0]], names[pair[1]]]

    new_names = copy.copy(names)
    del new_names[np.max(pair)]
    del new_names[np.min(pair)]
    new_names.append(namecount)
    newnamecount = namecount + 1

    parent_cell = names_to_cells[int(names[np.max(pair)])] + "," + names_to_cells[int(names[np.min(pair)])]
    new_names_to_cells = names_to_cells.copy() + [parent_cell]

    edges, norm_factor, choices = simulate_clt_sample_rec(
        P_new, greedy, c, new_names_to_cells, clade, choices_made, subtrees_cell_list, new_names, newnamecount, eps, delta, divide, coef, False, norm_factor
    )
    edges.append(new_edge)
    return edges, norm_factor, choices + [ind]

def clt_sample_rec(
    P,
    greedy,
    c,
    names_to_cells,
    clade,
    rng=None,
    names=None,
    namecount=None,
    eps=.1,
    delta=0.5,
    divide=False,
    coef=2,
    first=True,
    prior_prob=None,
    join_prob_matrix=None
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
    log = ""

    if len(P) > 1 and not first:
        log += "\t------- Next iteration --------" + "\n"
    # TODO make this faster by not recalculating
    if prior_prob is None:
        prior_prob = Decimal(1.0)
    if P.shape[0] == 1:
        return [], prior_prob, []
    if names is None:
        names = list(range(P.shape[0]))
        namecount = P.shape[0]

    if join_prob_matrix is None:

        def join_neg_priority(a, b):  # smaller values should be joined first
            return la.norm(a - b) - row_leafness_score(a, b) * coef
        
            # norm_a = la.norm(a)
            # norm_b = la.norm(b)
            # norm_max = max(norm_a, norm_b)
            # return (1 - coef) * la.norm((a / norm_max) - (b / norm_max)) - coef * cosine(a, b)
            # return la.norm(a - b) - np.dot(a,b) * coef

        dist = pairwise_distances(P, metric=join_neg_priority)  # O(n m^2)
        dist = dist.astype(np.float64)

        np.fill_diagonal(dist, np.inf)
        save = dist.copy()

        if divide:
            #divide to normalize to max:0 min: -eps
            dist = -dist
            dist = dist - max(dist.flat)
            dabs_max = 0
            for entry in dist.flat:
                if entry != float('-inf') and entry != float('inf') and abs(entry) > dabs_max:
                    dabs_max = abs(entry)

            if dabs_max != 0:
                dist = -dist * (eps / dabs_max)
            else:
                dist = -dist
        else:
            #change the base of the softmax exponentation
            dist = -dist
            dist = dist - max(dist.flat)
            dist = -dist * np.log(1 + eps)

        

        

        # This block adjusts c if dist/2c is too big for sotfmax.
        c_rec = c
        # John: I don't think this is necessary
        # for _ in range(10):
        #     sim = softmax(-dist / (2 * c_rec))
        #     if not np.any(np.isnan(sim)):
        #         break
        #     c_rec *= 2.0

        # John: cut off values below x%
        sim = softmax(-dist)
        # distcopy = dist.copy()
        for i in range(len(sim.flat)):
            # if sim.flat[i] < 0.02:
            #     distcopy.flat[i] = float('inf')
            if sim.flat[i] < 0.01:
                sim.flat[i] = 0
        if np.dot(sim.flat, sim.flat) == 0:
            sim = softmax(-dist)
        else:
            sim = sim / sum(sim.flat)
        # sim = softmax(-distcopy)
    else:
        # add new row to join_prob_matrix to get sim
        pass
    prob = sim
    if greedy:
        pair = np.unravel_index(np.argmax(prob), prob.shape)
    else:
        flat_probs = np.float64(prob.flat)
        ind = None
        pair = None
        if rng == None:
            np.random.seed(seed=None)
            ind = np.random.choice(len(flat_probs), p=flat_probs)
            pair = np.unravel_index(ind, prob.shape)
        else:
            ind = rng.choice(len(flat_probs), p=flat_probs)
            pair = np.unravel_index(ind, prob.shape)

        

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

        index_and_ps.append((prob.flat[i], cells1, cells2, nbg, dist.flat[i], save.flat[i]))
    index_and_ps = sorted(list(set(index_and_ps)), reverse=True)
    strings = []
    for i in range(len(index_and_ps)):
        temp = ""
        tup = index_and_ps[i]
        # if tup[0] * 2 < 0.00001:
        #     break
        if tup[1] == tup[2]: 
            pass
        else:
            temp += "\t" + tup[1] + " and " + tup[2] + ": p = " + str("{:.10f}".format(tup[0] * 2)) + " d = " + str("{:.10f}".format(tup[4])) + " s = " + str("{:.10f}".format(tup[5])) + " " + tup[3]
            strings.append(temp)
    # maxlen = len(max(strings, key=len))
    # strings = [s.ljust(maxlen) for s in strings]

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

    for i in range(25):
        if i < len(strings):
            log += strings[i] + "\n"

    # for s in strings:
    #     print(s)

    log += "\n\tSum of distribution = " + str(sum(prob.flat)) + "\n"
    
    
    log += "\tPair chosen: " + cells1 + " and " + cells2 + ": " + str("{:.10f}".format(prob.flat[ind] * 2)) + " " + nbg + "\n"
    
    # print(log)



    # if first:
    #     print(log)
    # if nbg == "BAD" and first:
    #     first = False
    #     print(log)


    

    # conversion from numpy.float128 to Decimal is not supported
    prior_prob = prior_prob * Decimal(np.float64(prob[pair]))
    P_new = np.delete(P, pair, axis=0)  # remove two rows
    # P_new = np.append(
    #     P_new, np.minimum(P[pair[0]], P[pair[1]]).reshape(1, -1), axis=0
    # )  # add one row that has only common ones
    # P_new = np.append(
    #     P_new, np.minimum(P[pair[0]], P[pair[1]]).reshape(1, -1), axis=0
    # )  # add one row that has only common ones

    new_row = delta * np.minimum(P[pair[0]], P[pair[1]]) + (1 - delta) * np.maximum(P[pair[0]], P[pair[1]])
    P_new = np.append(
        P_new, new_row.reshape(1, -1), axis=0
    )
    new_edge = [namecount, names[pair[0]], names[pair[1]]]

    new_names = copy.copy(names)
    del new_names[np.max(pair)]
    del new_names[np.min(pair)]
    new_names.append(namecount)
    newnamecount = namecount + 1

    parent_cell = names_to_cells[int(names[np.max(pair)])] + "," + names_to_cells[int(names[np.min(pair)])]
    new_names_to_cells = names_to_cells.copy() + [parent_cell]

    edges, prior_prob, choices = clt_sample_rec(
        P_new, greedy, c, new_names_to_cells, clade, rng, new_names, newnamecount, eps, delta, divide, coef, False, prior_prob
    )
    edges.append(new_edge)
    return edges, prior_prob, [ind] + choices 


def row_leafness_score(row_a, row_b):
    return np.sum(np.minimum(row_a, row_b))

def cosine(row_a, row_b):
    return np.dot(row_a, row_b) / (la.norm(row_a) * la.norm(row_b))
    
