#!/usr/bin/env python3
"""
A more 'real' MAGNN implementation on the URI dataset.

1) Loads user->recipe edges + other edges. 
2) Creates real adjacency structures for (U-R), (R-I), etc.
3) Defines metapath expansions: e.g. U->R and U->R->I->R. 
4) Does link prediction (train/val/test with negative sampling). 
5) Evaluates with both AUC/AP (pairwise) and ranking-based metrics (Hit@K, NDCG@K, Precision@K, MAP@K) using Leave-One-Out. 
"""

import os
import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import Adam

###############################################################################
# 1. DATA LOADER
###############################################################################
class URIDataLoader(object):
    """
    Loads the URI heterogeneous graph dataset. This includes:
      - Node types: [User, Recipe, Ingredient]
      - Edges: U-R, R-R, R-I, I-I
      - Also constructs adjacency lists for each node type, so we can do
        real metapath expansions (U->R->I->R, etc.)
      - Train/Val/Test splits for user->recipe edges with negative sampling.
    """

    def __init__(self, data_dir="data", device="cpu", seed=42):
        super(URIDataLoader, self).__init__()
        self.data_dir = data_dir
        self.device = device
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Basic stats
        self.num_users = 0
        self.num_recipes = 0
        self.num_ingredients = 0

        # Edges
        self.user_recipe_edges = []           # (user, recipe, rating)
        self.recipe_recipe_edges = []         # (r1, r2, sim)
        self.recipe_ingredient_edges = []     # (recipe, ingredient, usage_wt)
        self.ingredient_ingredient_edges = [] # (i1, i2, cooccur)

        # Train / val / test edges (pos + neg)
        self.train_pos = []
        self.train_neg = []
        # self.train_pos = self.train_pos[:10000]  # only 10k
        # self.train_neg = self.train_neg[:10000] # only 10k
        self.val_pos = []
        self.val_neg = []
        self.test_pos = []
        self.test_neg = []

        # Adjacency lists: We'll store them for real aggregator calls
        #   user->recipes adjacency (train-based)
        self.user2recipes = defaultdict(list)       # user-> [recipes user has in train]
        self.recipe2users = defaultdict(list)       # recipe-> [users in train]
        self.recipe2ingredients = defaultdict(list) # recipe-> [ingredients]
        self.ingredient2recipes = defaultdict(list) # ingredient-> [recipes]

        self._load_data()
        self._split_user_recipe_edges()

        # Build adjacency from train edges
        self._build_train_adjacency()

    def _load_data(self):
        """
        Loads from .pt files. Adjust as needed for your actual layout.
        """
        # 1) user->recipe edges
        user_recipe_data = torch.load(
            os.path.join(self.data_dir, "all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt")
        )
        ur_src, ur_dst, ur_wt = user_recipe_data[0]
        ur_src = np.array(ur_src, dtype=np.int32)
        ur_dst = np.array(ur_dst, dtype=np.int32)
        ur_wt  = np.array(ur_wt,  dtype=np.float32)
        self.user_recipe_edges = list(zip(ur_src, ur_dst, ur_wt))

        # 2) recipe->recipe
        r2r_data = torch.load(
            os.path.join(self.data_dir, "edge_r2r_src_and_dst_and_weight.pt")
        )
        print("Type of r2r_data:", type(r2r_data))
        print("Length of r2r_data:", len(r2r_data))
        print("Type of r2r_data[0]:", type(r2r_data[0]))

        if isinstance(r2r_data[0], (list, tuple)):
            print("Length of r2r_data[0]:", len(r2r_data[0]))
            print("First few entries:", r2r_data[0][:10])  # Print first 10 elements
        
        print("Type of r2r_data:", type(r2r_data))
        print("Length of r2r_data:", len(r2r_data))

        if isinstance(r2r_data, list) and len(r2r_data) == 3:
            print("Each part:")
            print("  r2r_data[0] length:", len(r2r_data[0]))
            print("  r2r_data[1] length:", len(r2r_data[1]))
            print("  r2r_data[2] length:", len(r2r_data[2]))
        else:
            print("Unexpected format:", r2r_data)
        if len(r2r_data) == 3:
            r2r_src, r2r_dst, r2r_wt = r2r_data
        else:
            raise ValueError(f"Expected r2r_data to have exactly 3 elements, but got {len(r2r_data)}")
        # r2r_src, r2r_dst, r2r_wt = r2r_data[0]
        r2r_src, r2r_dst, r2r_wt = r2r_data
        r2r_src = np.array(r2r_src, dtype=np.int32)
        r2r_dst = np.array(r2r_dst, dtype=np.int32)
        r2r_wt  = np.array(r2r_wt,  dtype=np.float32)
        self.recipe_recipe_edges = list(zip(r2r_src, r2r_dst, r2r_wt))

        # 3) recipe->ingredient
        r2i_data = torch.load(
            os.path.join(self.data_dir, "edge_r2i_src_dst_weight.pt")
        )
        # r2i_src, r2i_dst, r2i_wt = r2i_data[0]
        r2i_src, r2i_dst, r2i_wt = r2i_data
        r2i_src = np.array(r2i_src, dtype=np.int32)
        r2i_dst = np.array(r2i_dst, dtype=np.int32)
        r2i_wt  = np.array(r2i_wt,  dtype=np.float32)
        self.recipe_ingredient_edges = list(zip(r2i_src, r2i_dst, r2i_wt))

        # 4) ingredient->ingredient
        i2i_data = torch.load(
            os.path.join(self.data_dir, "edge_i2i_src_and_dst_and_weight.pt")
        )
        # i2i_src, i2i_dst, i2i_wt = i2i_data[0]
        i2i_src, i2i_dst, i2i_wt = i2i_data
        i2i_src = np.array(i2i_src, dtype=np.int32)
        i2i_dst = np.array(i2i_dst, dtype=np.int32)
        i2i_wt  = np.array(i2i_wt,  dtype=np.float32)
        self.ingredient_ingredient_edges = list(zip(i2i_src, i2i_dst, i2i_wt))

        # Suppose the dataset is known:
        self.num_users      = 7959
        self.num_recipes    = 68794
        self.num_ingredients= 8847

    def _split_user_recipe_edges(self, train_ratio=0.7, val_ratio=0.1):
        """
        Splits the user->recipe edges into train, val, test sets (pos only).
        Then negative sampling for each set.
        """
        edges = self.user_recipe_edges
        random.shuffle(edges)
        n_total = len(edges)
        n_train = int(n_total * train_ratio)
        n_val   = int(n_total * val_ratio)
        train_edges = edges[:n_train]
        val_edges   = edges[n_train : n_train + n_val]
        test_edges  = edges[n_train + n_val : ]

        # Build set for membership
        all_ur_set = set()
        for (u,r,w) in edges:
            all_ur_set.add((u,r))

        self.train_pos = train_edges
        self.val_pos   = val_edges
        self.test_pos  = test_edges

        self.train_neg = self._build_neg_pairs(train_edges, len(train_edges), all_ur_set)
        self.val_neg   = self._build_neg_pairs(val_edges,   len(val_edges),   all_ur_set)
        self.test_neg  = self._build_neg_pairs(test_edges,  len(test_edges),  all_ur_set)

    def _build_neg_pairs(self, pos_edges, n_samples, ur_set):
        """
        For each pos edge, sample 1 negative. Or any ratio you want.
        """
        # neg_edges = []
        # tried = 0
        # while len(neg_edges) < n_samples and tried < n_samples*100:
        #     u_rand = random.randint(0, self.num_users-1)
        #     r_rand = random.randint(0, self.num_recipes-1)
        #     if (u_rand, r_rand) not in ur_set:
        #         neg_edges.append((u_rand, r_rand, 0.0))
        #     tried += 1
        # return neg_edges

        # Simpler negative sampling approach (like random pairs that definitely do not overlap).
        neg_edges = []
        for (u, r, w) in pos_edges[:n_samples]:
            # sample exactly 1 negative
            while True:
                rand_r = random.randint(0, self.num_recipes - 1)
                if (u, rand_r) not in ur_set:
                    neg_edges.append((u, rand_r, 0.0))
                    break
        return neg_edges

    def _build_train_adjacency(self):
        """
        Build adjacency dictionaries for user->recipes (train only),
        recipe->ingredients, ingredient->recipes, etc.
        We'll need these for real metapath expansions in aggregator calls.
        """
        # user->recipes from train
        for (u, r, w) in self.train_pos:
            self.user2recipes[u].append(r)
            self.recipe2users[r].append(u)

        # recipe->ingredient
        for (r, i, w) in self.recipe_ingredient_edges:
            self.recipe2ingredients[r].append(i)
            self.ingredient2recipes[i].append(r)

    def get_train_data(self):
        return self.train_pos, self.train_neg

    def get_val_data(self):
        return self.val_pos, self.val_neg

    def get_test_data(self):
        return self.test_pos, self.test_neg

    def to(self, device):
        self.device = device


###############################################################################
# 2. MODEL COMPONENTS (REAL NEIGHBOR SAMPLING)
###############################################################################
class MetapathInstanceEncoder(nn.Module):
    """
    Encodes a single metapath instance by aggregating the node features along that path.
    """
    def __init__(self, input_dim, method="mean"):
        super().__init__()
        self.method = method
        self.input_dim = input_dim
        if method == "linear":
            self.fc = nn.Linear(input_dim, input_dim)
            nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        elif method.startswith("rotate"):
            pass  # omitted for brevity

    def forward(self, node_feats):
        """
        node_feats: (L x d) or a list of node feature vectors
        """
        if isinstance(node_feats, list):
            node_feats = torch.stack(node_feats, dim=0)  # (L, d)
        if self.method == "mean":
            return node_feats.mean(dim=0)
        elif self.method == "linear":
            return self.fc(node_feats.mean(dim=0))
        elif self.method.startswith("rotate"):
            return node_feats.mean(dim=0)  # placeholder
        else:
            return node_feats.mean(dim=0)


class MAGNNIntraMetapathAggregator(nn.Module):
    """
    Given a target node's embedding + the expansions along a certain metapath,
    encodes each "instance" (a path from target->...->neighbor) with a MetapathInstanceEncoder,
    then aggregates them by multi-head attention.
    """
    def __init__(self, input_dim, num_heads=4, encoder_method="mean"):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.instance_encoder = MetapathInstanceEncoder(input_dim, method=encoder_method)

        # For multi-head attention
        self.attn_fc = nn.Parameter(torch.Tensor(num_heads, 2 * input_dim))
        nn.init.xavier_uniform_(self.attn_fc, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, target_feat, list_of_paths):
        """
        list_of_paths: each path is a list/tensor of node features (including target?)
        """
        # encode each path
        instance_reps = []
        for path_feats in list_of_paths:
            rep = self.instance_encoder(path_feats)
            instance_reps.append(rep)

        if len(instance_reps) == 0:
            # no neighbors
            return torch.zeros(self.num_heads*self.input_dim, device=target_feat.device)

        instance_reps = torch.stack(instance_reps, dim=0)  # (N, d)
        N = instance_reps.size(0)

        # multi-head attention
        # e = a_k^T [ target || instance_rep ]
        target_expand = target_feat.unsqueeze(0).expand(N, -1)      # (N, d)
        cat = torch.cat([target_expand, instance_reps], dim=1)      # (N, 2d)
        # replicate for heads
        cat_expand = cat.unsqueeze(0).expand(self.num_heads, N, 2*self.input_dim)  # (num_heads, N, 2d)
        # attn_fc: (num_heads, 2d)
        e = torch.bmm(cat_expand, self.attn_fc.unsqueeze(2)).squeeze(-1)  # (num_heads, N)
        e = self.leaky_relu(e)
        alpha = F.softmax(e, dim=1)  # (num_heads, N)

        # Weighted sum
        instance_reps_expand = instance_reps.unsqueeze(0).expand(self.num_heads, N, self.input_dim)
        out_per_head = torch.bmm(alpha.unsqueeze(1), instance_reps_expand).squeeze(1)  # (num_heads, d)

        out = out_per_head.view(self.num_heads * self.input_dim)
        return out


class MAGNNInterMetapathAggregator(nn.Module):
    """
    Combines multiple metapath embeddings with an attention mechanism.
    """
    def __init__(self, input_dim, num_heads, attn_vec_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attn_vec_dim = attn_vec_dim

        # after intra-agg we have (num_heads*input_dim)
        self.fc = nn.Linear(num_heads*input_dim, attn_vec_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.attn_vec = nn.Linear(attn_vec_dim, 1, bias=False)
        nn.init.xavier_normal_(self.attn_vec.weight, gain=1.414)

    def forward(self, list_of_metapath_embs):
        """
        Each element: (num_heads*input_dim)
        We'll do alpha_i = softmax( q^T tanh( W * h_i ) )
        """
        if len(list_of_metapath_embs) == 0:
            return torch.zeros(self.num_heads*self.input_dim, device=self.fc.weight.device)

        H = torch.stack(list_of_metapath_embs, dim=0)  # (M, num_heads*input_dim)
        # shape (M, attn_vec_dim)
        wh = torch.tanh(self.fc(H))
        # shape (M, 1)
        alphas = self.attn_vec(wh)
        alphas = F.softmax(alphas, dim=0)
        out = (H * alphas).sum(dim=0)  # (num_heads*input_dim)
        return out


class MAGNNModel(nn.Module):
    """
    A more realistic MAGNN-based link-prediction model, with:
      - Real expansions for U->R and U->R->I->R on user side,
        and R->U / R->I->R on recipe side, etc.
      - We store an embedding matrix for users/recipes (and optionally ingredients).
      - We do aggregator calls with actual neighbors from the adjacency dictionary.
    """
    def __init__(self, num_user, num_recipe, 
                 user2recipes, recipe2ingredients,
                 input_dim=64, num_heads=4, encoder_method="mean", device="cpu"):
        super().__init__()
        self.num_user = num_user
        self.num_recipe = num_recipe
        self.user2recipes = user2recipes
        self.recipe2ingredients = recipe2ingredients
        self.device = device
        self.input_dim = input_dim
        self.num_heads = num_heads

        # Define the "intra" aggregators for user side:
        self.agg_UR   = MAGNNIntraMetapathAggregator(input_dim, num_heads, encoder_method)  # user->recipe
        self.agg_URIR = MAGNNIntraMetapathAggregator(input_dim, num_heads, encoder_method)  # user->recipe->ingredient->recipe

        # "inter" aggregator
        self.inter_agg_user = MAGNNInterMetapathAggregator(input_dim, num_heads)

        # For recipe side, we can define symmetric or different aggregator calls
        self.agg_RU   = MAGNNIntraMetapathAggregator(input_dim, num_heads, encoder_method)
        self.agg_RIR  = MAGNNIntraMetapathAggregator(input_dim, num_heads, encoder_method)
        self.inter_agg_recipe = MAGNNInterMetapathAggregator(input_dim, num_heads)

        # Embeddings
        self.user_emb = nn.Embedding(num_user, input_dim)
        self.recipe_emb = nn.Embedding(num_recipe, input_dim)
        nn.init.xavier_normal_(self.user_emb.weight, gain=1.414)
        nn.init.xavier_normal_(self.recipe_emb.weight, gain=1.414)

        self.num_ingredient = 8847  # (or pass it in from data_loader)
        self.ingredient_emb = nn.Embedding(self.num_ingredient, input_dim)
        nn.init.xavier_normal_(self.ingredient_emb.weight, gain=1.414)

        self.to(device)

    def forward_user(self, u_id):
        """
        Build user embedding by two metapaths: UR, URIR.
        """
        # Base feature for user
        user_feat = self.user_emb(torch.tensor([u_id], device=self.device)).squeeze(0)

        # Expand UR neighbors
        # For aggregator, each neighbor path is [ user_feat, recipe_feat ]
        # so instance = 2 nodes
        ur_paths = []
        recipes = self.user2recipes[u_id] if u_id in self.user2recipes else []
        for r in recipes:
            r_feat = self.recipe_emb(torch.tensor([r], device=self.device)).squeeze(0)
            ur_paths.append( torch.stack([user_feat, r_feat], dim=0) )

        UR_out = self.agg_UR(user_feat, ur_paths)

        # Expand URIR neighbors
        # For aggregator, each neighbor path is [ user_feat, r_feat, i_feat, r2_feat ]
        # but let's limit ourselves to a small sample so it won't blow up.
        urir_paths = []
        for r in recipes[:20]:  # sample up to 20, 5
            r_feat = self.recipe_emb(torch.tensor([r], device=self.device)).squeeze(0)
            # gather ingredients
            ingredients = self.recipe2ingredients[r] if r in self.recipe2ingredients else []
            for i_id in ingredients[:5]:  # sample up to 5, 2
                # recipe->ingredient->(some-other-recipe)? 
                # The original path is U->R->I->R. We can pick the same r or any r that uses ingredient i.
                i_feat = self.ingredient_emb(torch.tensor([i_id], device=self.device)).squeeze(0)
                # if we had an ingredient embedding
                # For simplicity, skip or set to zero vector, or define self.ingredient_emb
                # Then pick r again or a random recipe that uses i
                r2 = r  # simplistic approach
                r2_feat = self.recipe_emb(torch.tensor([r2], device=self.device)).squeeze(0)
                path_feats = [user_feat, r_feat, i_feat, r2_feat]
                urir_paths.append(torch.stack(path_feats, dim=0))

        URIR_out = self.agg_URIR(user_feat, urir_paths)

        # Inter aggregator
        user_final = self.inter_agg_user([UR_out, URIR_out])
        return user_final

    def forward_recipe(self, r_id):
        """
        Build recipe embedding by two metapaths: R->U, R->I->R
        """
        recipe_feat = self.recipe_emb(torch.tensor([r_id], device=self.device)).squeeze(0)

        # R->U aggregator
        # For aggregator, each path is [ recipe, user ]
        # We didn't store recipe->users adjacency above except in user2recipes
        # so let's skip or fill if we had that
        # (We can build recipe2users if we want symmetrical adjacency in _build_train_adjacency)
        # For demonstration, let's assume we have recipe2users (not shown above).
        # We'll pass 0 neighbors if we don't have it.
        # (We do show recipe2users in the code, so let's use it.)
        # We'll do something symmetrical:
        # recipe2users is not directly stored above in the final code, let's do it if we want:
        # -> We can replicate user2recipes in reverse.
        # We'll skip it for brevity or define it quickly:
        RU_paths = []
        # Suppose we have self.recipe2users from data_loader in constructor (passed in)
        if hasattr(self, 'recipe2users'):
            users = self.recipe2users.get(r_id, [])
        else:
            users = []
        for u_id in users[:20]:  # sample, 5, 20
            u_feat = self.user_emb(torch.tensor([u_id], device=self.device)).squeeze(0)
            RU_paths.append(torch.stack([recipe_feat, u_feat], dim=0))

        RU_out = self.agg_RU(recipe_feat, RU_paths)

        # R->I->R aggregator
        # path: [ recipe, i_feat, recipe2 ], but we’ll do a short version
        RIR_paths = []
        ings = self.recipe2ingredients.get(r_id, [])
        for i_id in ings[:5]: # sample, 2, 5
            i_feat = torch.zeros(self.input_dim, device=self.device)  # or self.ingredient_emb(...) if you had it
            # pick r2 as same r or some other
            r2 = r_id  
            r2_feat = self.recipe_emb(torch.tensor([r2], device=self.device)).squeeze(0)
            RIR_paths.append(torch.stack([recipe_feat, i_feat, r2_feat], dim=0))

        RIR_out = self.agg_RIR(recipe_feat, RIR_paths)

        recipe_final = self.inter_agg_recipe([RU_out, RIR_out])
        return recipe_final

    def predict_score(self, u_id, r_id):
        hu = self.forward_user(u_id)
        hr = self.forward_recipe(r_id)
        return torch.dot(hu, hr)

    def forward(self, user_ids, recipe_ids):
        out = []
        for u, r in zip(user_ids, recipe_ids):
            s = self.predict_score(int(u), int(r))
            out.append(s)
        return torch.stack(out, dim=0)


###############################################################################
# 3. TRAINING/INFERENCE WITH RANK-BASED METRICS
###############################################################################
def precision_at_k(recommended, ground_truth, k=10):
    if not recommended:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k).intersection(set(ground_truth)))
    return hits / float(k)

def hit_rate_at_k(recommended, ground_truth, k=10):
    rec_k = recommended[:k]
    hits = set(rec_k).intersection(set(ground_truth))
    return 1.0 if len(hits) > 0 else 0.0

def _dcg_at_k(recommended, ground_truth, k=10):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i+2)
    return dcg

def ndcg_at_k(recommended, ground_truth, k=10):
    dcg = _dcg_at_k(recommended, ground_truth, k)
    # best DCG is if the ground_truth is ranked at top
    ideal_list = list(ground_truth)
    idcg = _dcg_at_k(ideal_list, ground_truth, min(k, len(ground_truth)))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def map_at_k(recommended, ground_truth, k=10):
    # average precision
    # if multiple ground-truth items, we sum precision at each relevant item rank, then average
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            hits += 1
            sum_precisions += hits / (i+1.0)
    if hits == 0:
        return 0.0
    return sum_precisions / hits


class Trainer(object):
    """
    Trainer with negative sampling (binary cross-entropy) and also 
    advanced ranking evaluation with leave-one-out if desired.
    """
    def __init__(self, model, data_loader, device="cpu", lr=0.001, weight_decay=1e-4):
        self.model = model
        self.loader = data_loader
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, train_pos, train_neg, batch_size=512):
        self.model.train()
        # pairs_pos = train_pos[:10000]  # limit to 10k for speed
        # pairs_neg = train_neg[:10000]  # limit to 10k for speed
        random.shuffle(pairs_pos)
        random.shuffle(pairs_neg)

        def bce_loss_fn(logits, labels):
            offset = 0.01
            return F.binary_cross_entropy_with_logits(logits + offset, labels)


        n = len(pairs_pos)
        n_batches = math.ceil(n / batch_size)
        total_loss = 0.0

        for i in range(n_batches):
            start = i*batch_size
            end = min(start+batch_size, n)
            pos_batch = pairs_pos[start:end]
            neg_batch = pairs_neg[start:end]

            user_ids, recipe_ids, labels = [], [], []
            for (u, r, w) in pos_batch:
                user_ids.append(u)
                recipe_ids.append(r)
                labels.append(1.0)
            for (u, r, w) in neg_batch:
                user_ids.append(u)
                recipe_ids.append(r)
                labels.append(0.0)

            user_ids = torch.tensor(user_ids, device=self.device)
            recipe_ids = torch.tensor(recipe_ids, device=self.device)
            labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            logits = self.model.forward(user_ids, recipe_ids)
            loss = bce_loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            # Print batch progress (flush immediately)
            print(f"Batch {i+1}/{n_batches} - Loss: {loss.item():.4f}  Running Avg: {total_loss/(i+1):.4f}")
            sys.stdout.flush()

        return total_loss / n_batches

    def evaluate_auc_ap(self, pos_edges, neg_edges, batch_size=512):
        """
        Returns (auc, ap). Pairwise classification approach.
        """
        self.model.eval()
        user_pos = [p[0] for p in pos_edges]
        item_pos = [p[1] for p in pos_edges]
        user_neg = [p[0] for p in neg_edges]
        item_neg = [p[1] for p in neg_edges]

        labels = np.array([1]*len(pos_edges) + [0]*len(neg_edges), dtype=np.float32)

        # compute scores
        scores = []
        def batched_eval(u_list, r_list):
            local_scores = []
            ndata = len(u_list)
            nb = math.ceil(ndata / batch_size)
            with torch.no_grad():
                for bi in range(nb):
                    st = bi*batch_size
                    en = min(st+batch_size, ndata)
                    uu = torch.tensor(u_list[st:en], device=self.device)
                    rr = torch.tensor(r_list[st:en], device=self.device)
                    s = self.model.forward(uu, rr).cpu().numpy()
                    local_scores.append(s)
            return np.concatenate(local_scores, axis=0)

        scores_pos = batched_eval(user_pos, item_pos)
        scores_neg = batched_eval(user_neg, item_neg)
        all_scores = np.concatenate([scores_pos, scores_neg], axis=0)
        auc = roc_auc_score(labels, all_scores)
        ap  = average_precision_score(labels, all_scores)
        return auc, ap

    def evaluate_ranking_loo(self, test_pos, K=10, num_neg=100):
        """
        Leave-One-Out approach: For each user in test_pos, pick exactly one test edge
        (user->item) as the ground-truth. Then sample 'num_neg' negative items for that user,
        rank them all, and compute Hit@K, NDCG@K, etc. A standard approach in recommender eval.

        test_pos: list of (u, r, w). If user has multiple test items, we handle each item individually.
        """
        self.model.eval()
        # Group test items by user
        user2items = defaultdict(list)
        for (u, r, w) in test_pos:
            user2items[u].append(r)

        hr_list, ndcg_list, prec_list, map_list = [], [], [], []

        all_users = sorted(user2items.keys())
        # We'll do a single LOO per test item if the user has multiple test edges
        # (some do a single item, others do multiple). We'll handle them in a small loop.

        # We'll need a global set of train edges so we don't sample positives as negatives
        train_UR = set()
        for (u,r,_) in self.loader.train_pos:
            train_UR.add((u,r))

        for user in all_users:
            test_recipes = user2items[user]  # may be multiple
            for gt_item in test_recipes:
                # Sample num_neg negative
                neg_items = []
                tried = 0
                while len(neg_items) < num_neg and tried < num_neg*100:
                    candidate = random.randint(0, self.loader.num_recipes-1)
                    if (user, candidate) not in train_UR and candidate not in test_recipes:
                        neg_items.append(candidate)
                    tried += 1

                # now we have 1 positive (gt_item) + num_neg negatives
                # let's compute scores
                candidates = [gt_item] + neg_items
                user_ids = torch.tensor([user]*len(candidates), device=self.device)
                item_ids = torch.tensor(candidates, device=self.device)
                with torch.no_grad():
                    scores = self.model.forward(user_ids, item_ids).cpu().numpy()

                # sort candidates by score descending
                sorted_idx = np.argsort(scores)[::-1]
                ranked_items = [candidates[i] for i in sorted_idx]

                # compute metrics
                ground_truth = [gt_item]
                hr = hit_rate_at_k(ranked_items, ground_truth, K)
                nd = ndcg_at_k(ranked_items, ground_truth, K)
                pc = precision_at_k(ranked_items, ground_truth, K)
                mp = map_at_k(ranked_items, ground_truth, K)

                hr_list.append(hr)
                ndcg_list.append(nd)
                prec_list.append(pc)
                map_list.append(mp)

        # average
        hr_ = np.mean(hr_list)
        nd_ = np.mean(ndcg_list)
        pr_ = np.mean(prec_list)
        mp_ = np.mean(map_list)
        return hr_, nd_, pr_, mp_

###############################################################################
# 4. MAIN
###############################################################################
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    sys.stdout.flush()

    # Load data
    data_loader = URIDataLoader(data_dir="data", device=device, seed=42)
    print("Users:", data_loader.num_users, "Recipes:", data_loader.num_recipes)
    print("Train data size:", len(data_loader.train_pos))
    sys.stdout.flush()

    train_pos, train_neg = data_loader.get_train_data()
    val_pos, val_neg = data_loader.get_val_data()
    test_pos, test_neg = data_loader.get_test_data()

    # Build model
    model = MAGNNModel(
        num_user=data_loader.num_users,
        num_recipe=data_loader.num_recipes,
        user2recipes=data_loader.user2recipes,     # adjacency
        recipe2ingredients=data_loader.recipe2ingredients,
        input_dim=64,
        num_heads=4,
        encoder_method="mean",
        device=device
    )

    trainer = Trainer(model, data_loader, device=device, lr=0.001, weight_decay=1e-5)

    # Training loop
    best_val_auc = 0.0
    patience = 5
    cur_patience = 0
    best_state = None
    max_epochs = 20 #10

    
    for ep in range(max_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {ep} start:")
        sys.stdout.flush()

        trn_loss = trainer.train_epoch(train_pos, train_neg, batch_size=512) # 256, 64
        val_auc, val_ap = trainer.evaluate_auc_ap(val_pos, val_neg)
        
        epoch_time = time.time() - epoch_start  # Elapsed time for epoch

        print(f"Epoch {ep} finished: Loss={trn_loss:.4f}, Val AUC={val_auc:.4f}, AP={val_ap:.4f}, Time={epoch_time:.2f}s")
        sys.stdout.flush()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            cur_patience = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            # Incrementally save the model
            checkpoint_path = f"BaseMAGNN_checkpoint_epoch_{ep}.pt"
            torch.save(best_state, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            sys.stdout.flush()
        else:
            cur_patience += 1
            if cur_patience >= patience:
                print("Early stopping triggered.")
                sys.stdout.flush()
                break

    # Load best model if available
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate final on test set (AUC, AP)
    test_auc, test_ap = trainer.evaluate_auc_ap(test_pos, test_neg)
    print(f"[Test Pairwise Classification] AUC={test_auc:.4f}, AP={test_ap:.4f}")

    # Evaluate with ranking-based LOO
    # Only do this if you'd like top-K metrics for a subset of test users
    # We'll do K=5, 100 negative samples
    hr, ndcg, prec, map_ = trainer.evaluate_ranking_loo(test_pos, K=5, num_neg=100)
    print(f"[Test Ranking LOO @5] HitRate={hr:.4f}, NDCG={ndcg:.4f}, Precision={prec:.4f}, MAP={map_:.4f}")

    # Run model in evaluation mode
    model.eval()
    user_ids = torch.tensor([0, 1, 2], device=device)  # Test with some user IDs
    recipe_ids = torch.tensor([10, 20, 30], device=device)  # Test with some recipes
    scores = model.forward(user_ids, recipe_ids)
    print("Sample predictions:", scores)

if __name__ == "__main__":
    main()

