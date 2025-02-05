import numpy as np
import heapq


def precision_at_k(r, k):
    r = np.asarray(r)[:k]
    return np.mean(r)

def dcg_at_k(r, k, method=0):
    # Change np.asfarray to np.asarray
    r = np.asarray(r, dtype=np.float64)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def hit_at_k(r, k):
    r = np.array(r)[:k]
    return 1.0 if np.sum(r) > 0 else 0.0

def average_precision_at_k(r, Ks):
    # interpret r as boolean array
    r = np.asarray(r) != 0
    out = []
    for k in Ks:
        # For each rank i up to k, if r[i] == 1, we compute precision @ i+1, else ignore
        precisions = [precision_at_k(r, i+1) for i in range(k) if r[i]]
        if len(precisions) == 0:
            precisions = [0]
        out.append(np.mean(precisions))
    return np.array(out)

def get_map_at_k(rs, Ks):
    out = np.zeros(len(Ks))
    for r in rs:
        out += average_precision_at_k(r, Ks) / len(rs)
    return out

def get_ranklist_for_one_user(user_poss, user_negs, Ks):
    """
    Combine positive & negative scores into one dict, take the top K_max,
    produce a binary list (r) indicating 1 if the item is positive, else 0.
    """
    n_pos = len(user_poss)
    n_neg = len(user_negs)
    item_scores = {}
    # assign positive items an index [0..n_pos-1], negative items [n_pos..n_pos+n_neg-1]
    for i in range(n_pos):
        item_scores[i] = user_poss[i]
    for i in range(n_neg):
        item_scores[i + n_pos] = user_negs[i]

    K_max = max(Ks)
    topk_items = heapq.nlargest(K_max, item_scores, key=item_scores.get)
    r = [1 if i < n_pos else 0 for i in topk_items]
    return r

def get_performance_one_user(user_poss, user_negs, Ks):
    r = get_ranklist_for_one_user(user_poss, user_negs, Ks)
    precision, ndcg, hit_ratio = [], [], []
    for K in Ks:
        precision.append(precision_at_k(r, K))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))
    return {'precision': np.array(precision),
            'ndcg': np.array(ndcg),
            'hit_ratio': np.array(hit_ratio)}, r

def get_performance_all_users(user2pos_score_dict, user2neg_score_dict, Ks):
    all_result = {'hit_ratio': np.zeros(len(Ks)),
                  'ndcg': np.zeros(len(Ks)),
                  'precision': np.zeros(len(Ks))}
    rs = []
    n_test_users = len(user2pos_score_dict)
    for user in user2pos_score_dict.keys():
        user_pos_score = user2pos_score_dict[user]
        user_neg_score = user2neg_score_dict.get(user, [])  # safe-get
        one_result, r = get_performance_one_user(user_pos_score, user_neg_score, Ks)
        all_result['hit_ratio'] += one_result['hit_ratio']/n_test_users
        all_result['ndcg'] += one_result['ndcg']/n_test_users
        all_result['precision'] += one_result['precision']/n_test_users
        rs.append(r)
    MAP = get_map_at_k(rs, Ks)
    all_result['MAP'] = MAP
    return all_result