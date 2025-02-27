import torch
import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import normalize
import faiss
from collections import Counter


def load_split_data_80_10_10(dataset_folder="data/"):
    """
    Loads user–recipe interactions, creates a sparse matrix,
    and splits the interactions into train, validation, and test (80/10/10)
    on a per-user basis, *forcing at least one overlap item* per user
    if the user has >=2 interactions.

    Returns:
        train_set (csr_matrix), val_set (csr_matrix), test_set (csr_matrix)
    """
    # 1. Load the user–recipe edge data
    all_u2r_src_dst_weight = torch.load(
        os.path.join(dataset_folder, "all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt")
    )
    all_u2r_src, all_u2r_dst, all_u2r_weight = all_u2r_src_dst_weight[0]

    # 2. Convert to NumPy (for constructing the CSR/LIL matrix)
    all_u2r_src = np.array(all_u2r_src, dtype=np.int32)
    all_u2r_dst = np.array(all_u2r_dst, dtype=np.int32)
    all_u2r_weight = np.array(all_u2r_weight, dtype=np.float32)

    # 3. Define total number of users and recipes
    num_users = 7959
    num_recipes = 68794

    # 4. Create a sparse user–recipe matrix of shape (num_users, num_recipes)
    full_mat = csr_matrix(
        (all_u2r_weight, (all_u2r_src, all_u2r_dst)),
        shape=(num_users, num_recipes),
        dtype=np.float32
    )

    # 5. Prepare LIL matrices (more efficient for row-by-row modification)
    train_lil = lil_matrix(full_mat.shape, dtype=np.float32)
    val_lil   = lil_matrix(full_mat.shape, dtype=np.float32)
    test_lil  = lil_matrix(full_mat.shape, dtype=np.float32)

    # 6. Split interactions per user
    for user_id in range(num_users):
        user_row = full_mat[user_id]
        recipe_indices = user_row.indices
        recipe_values  = user_row.data
        if len(recipe_indices) == 0:
            continue

        # (a) Shuffle the user’s interactions
        perm = np.random.permutation(len(recipe_indices))
        recipe_indices = recipe_indices[perm]
        recipe_values  = recipe_values[perm]

        # (b) If the user has 2+ interactions, force exactly 1 overlap item
        #     into both train & test. (Pick it randomly.)
        overlap_item = None
        overlap_val  = None
        if len(recipe_indices) >= 2:
            overlap_item = recipe_indices[0]
            overlap_val  = recipe_values[0]
            # Remove it from the "pool" for normal 80/10/10 splitting
            recipe_indices = recipe_indices[1:]
            recipe_values  = recipe_values[1:]
            # Put that overlap item in both train & test
            train_lil[user_id, overlap_item] = overlap_val
            test_lil[user_id, overlap_item]  = overlap_val

        # (c) Now do the standard 80/10/10 split on the remainder
        n_total = len(recipe_indices)
        if n_total > 0:
            n_train = int(0.8 * n_total)
            n_val   = int(0.1 * n_total)
            n_test  = n_total - n_train - n_val

            train_recipes = recipe_indices[:n_train]
            train_vals    = recipe_values[:n_train]

            val_recipes   = recipe_indices[n_train : n_train + n_val]
            val_vals      = recipe_values[n_train : n_train + n_val]

            test_recipes  = recipe_indices[n_train + n_val:]
            test_vals     = recipe_values[n_train + n_val:]

            # (d) Assign to LIL
            if len(train_recipes) > 0:
                train_lil[user_id, train_recipes] = train_vals
            if len(val_recipes) > 0:
                val_lil[user_id, val_recipes]     = val_vals
            if len(test_recipes) > 0:
                test_lil[user_id, test_recipes]   = test_vals

    # 7. Convert LIL → CSR once at the end (avoiding repeated structure changes)
    train_csr = train_lil.tocsr()
    val_csr   = val_lil.tocsr()
    test_csr  = test_lil.tocsr()

    # 8. (Optional) row-wise normalize the training set
    #    Then back to CSR (this helps for k-NN user-based approaches)
    train_csr_normed = normalize(train_csr, norm='l2', axis=1)
    train_csr_normed = csr_matrix(train_csr_normed, dtype=np.float32)

    return train_csr_normed, val_csr, test_csr

def main():
    # 1. Load data (train, val, test) using the updated function
    train_set, val_set, test_set = load_split_data_80_10_10("data/")
    print("Data loaded. Shapes:")
    print("  Train:", train_set.shape, "nnz =", train_set.nnz)
    print("  Val:", val_set.shape,   "nnz =", val_set.nnz)
    print("  Test:", test_set.shape,  "nnz =", test_set.nnz)

    # 2. Convert to CSR (already is, but let's ensure)
    train_set_sparse = csr_matrix(train_set)
    val_set_sparse   = csr_matrix(val_set)
    test_set_sparse  = csr_matrix(test_set)

    # 3. Identify overlap in users (users who have ANY train interactions AND ANY test interactions)
    train_users = set(np.where(train_set_sparse.getnnz(axis=1) > 0)[0])
    test_users  = set(np.where(test_set_sparse.getnnz(axis=1) > 0)[0])
    common_users = sorted(list(train_users & test_users))

    # If you want to ensure only the subset of users who exist in both splits,
    # slice the matrix by `common_users`.
    train_set_sparse = train_set_sparse[common_users, :]
    test_set_sparse  = test_set_sparse[common_users, :]

    # 4. Build a dense representation for FAISS (inner product ~ cosine sim)
    num_users   = train_set_sparse.shape[0]
    num_recipes = train_set_sparse.shape[1]

    train_set_dense = train_set_sparse.toarray()  # row-wise normalized already
    # Replace any zero rows with small random noise to avoid index issues
    zero_rows = (train_set_dense.sum(axis=1) == 0)
    if np.any(zero_rows):
        train_set_dense[zero_rows] = np.random.uniform(
            low=0.001, high=0.01, size=(zero_rows.sum(), num_recipes)
        )

    # 5. Build FAISS index
    index = faiss.IndexFlatIP(num_recipes)  # inner product
    index.add(train_set_dense)  # add user vectors to index

    # 6. Define k-NN recommendation function
    def recommend_recipes_train(user_id, k=5):
        """
        Simple nearest-user approach: 
        1) Find top similar users via FAISS 
        2) Aggregate their recipes 
        3) Return top-k by frequency
        """
        if user_id < 0 or user_id >= num_users:
            raise ValueError("User ID out of range for the train/test overlap subset.")

        # Get user's seen recipes
        seen_recipes = set(train_set_sparse[user_id].indices)
        if len(seen_recipes) == 0:
            return []

        # Search nearest neighbors (top 20)
        n_search = min(20, num_users)
        query_vec = train_set_dense[user_id].reshape(1, -1)
        _, nearest_users = index.search(query_vec, n_search)
        # exclude self
        nearest_users = nearest_users.flatten()[1:]  

        recommended_recipes = []
        for sim_user in nearest_users:
            user_interactions = train_set_sparse[sim_user].indices
            recommended_recipes.extend(user_interactions)

        # Count frequency, return top-k
        recipe_counter = Counter(recommended_recipes)
        sorted_recipes = [r for r, _count in recipe_counter.most_common()]

        return sorted_recipes[:k]

    # 7. A simple LOO approach: pick 1 test item that's also in training for that user
    def leave_one_out_cv_test(user_local_id):
        test_items = set(test_set_sparse[user_local_id].indices)
        train_items = set(train_set_sparse[user_local_id].indices)
        valid_test_items = test_items & train_items

        if len(valid_test_items) == 0:
            return None
        # pick any
        test_item = np.random.choice(list(valid_test_items))
        return test_item

    # Define metrics inline (as lambdas) to avoid additional "def" statements.
    precision_at_k = lambda recommended, ground_truth, k: (
        len(set(recommended[:k]) & set(ground_truth)) / float(k) if ground_truth else 0.0
    )

    hit_rate_at_k = lambda recommended, ground_truth, k: (
        1.0 if set(recommended[:k]) & set(ground_truth) else 0.0
    )

    def _dcg_k(recommended, ground_truth, k):
        """
        Internal helper for calculating DCG. We define it as a small local function
        to keep the code block straightforward. This isn't a separate top-level `def`.
        """
        dcg_val = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                # Position i => rank i+1
                # So the contribution is 1/log2(i+2)
                dcg_val += 1.0 / np.log2(i + 2)
        return dcg_val

    ndcg_at_k = lambda recommended, ground_truth, k: (
        _dcg_k(recommended, ground_truth, k) /
        (_dcg_k(sorted(ground_truth), ground_truth, k) or 1.0)  # Avoid /0
    )

    def map_at_k(recommended, ground_truth, k):
        """
        Compute Mean Average Precision@K (MAP@K).
        In a LOO scenario, we usually have only 1 ground-truth item.
        If we don't hit that item in the top-K, we get an empty hits list => mean([]) => NaN.
        We'll return 0.0 if no hits occur.
        """
        if not ground_truth:
            return 0.0  # if there's no ground truth, map is 0

        hits_positions = []
        # Go through the recommended items up to k
        for j in range(min(k, len(recommended))):
            if recommended[j] in ground_truth:
                # 'j+1' is the rank position
                prec_j = precision_at_k(recommended, ground_truth, j + 1)
                hits_positions.append(prec_j)

        # If we never hit the ground-truth item => hits_positions is empty
        return np.mean(hits_positions) if hits_positions else 0.0

    np.random.seed(42)
    if num_users == 0:
        print("No common users with train/test data. Exiting.")
        return

    # We'll pick 10 random users (or fewer if num_users < 10)
    random_users = np.random.choice(num_users, size=min(10, num_users), replace=False)
    k_eval = 5

    # Simple 'Hit@K' tracking from your original snippet
    hits = 0
    total_users_evaluated = 0

    # For the new metrics, we collect them in lists
    precision_list = []
    hitrate_list   = []
    ndcg_list      = []
    map_list       = []

    for local_uid in random_users:
        test_item = leave_one_out_cv_test(local_uid)
        if test_item is None:
            # No suitable test item for this user
            continue

        # We have exactly one test item for LOO, so ground truth = [test_item]
        ground_truth = [test_item]

        # Get top-K recommended
        recommended = recommend_recipes_train(local_uid, k=k_eval)
        total_users_evaluated += 1
        
        # ========== Original Hit@K Computation ==========
        if test_item in recommended:
            hits += 1
        
        # ========== New Metrics Computation ==========
        # We pass the entire recommended list (up to k items) 
        # plus the single ground-truth item in a list
        prec = precision_at_k(recommended, ground_truth, k_eval)
        hr   = hit_rate_at_k(recommended, ground_truth, k_eval)
        ndcg = ndcg_at_k(recommended, ground_truth, k_eval)
        map_ = map_at_k(recommended, ground_truth, k_eval)
        
        precision_list.append(prec)
        hitrate_list.append(hr)
        ndcg_list.append(ndcg)
        map_list.append(map_)

    # Final aggregated results
    if total_users_evaluated > 0:
        # Original Hit@K
        print(f"Evaluated on {total_users_evaluated} users.")
        print(f"Hit@{k_eval} = {hits / total_users_evaluated:.2f}")

        # Our new metrics
        avg_precision = np.mean(precision_list)
        avg_hitrate   = np.mean(hitrate_list)
        avg_ndcg      = np.mean(ndcg_list)
        avg_map       = np.mean(map_list)

        print(f"Precision@{k_eval}: {avg_precision:.4f}")
        print(f"Hit Rate@{k_eval}:  {avg_hitrate:.4f}")
        print(f"NDCG@{k_eval}:      {avg_ndcg:.4f}")
        print(f"MAP@{k_eval}:       {avg_map:.4f}")
    else:
        print("No suitable test items found among random users. No evaluation performed.")

    # 9. Optional debug
    def debug_evaluation_issues():
        if num_users == 0:
            return

        random_user = np.random.randint(num_users)
        test_items = set(test_set_sparse[random_user].indices)
        train_items = set(train_set_sparse[random_user].indices)
        valid_test_items = test_items & train_items
        
        print(f"\n--- DEBUG for local user {random_user} ---")
        print(f"Train items: {len(train_items)} | Test items: {len(test_items)}")

        if not valid_test_items:
            print("No test item is in training set; can't be recommended.")
            return
        
        test_item = np.random.choice(list(valid_test_items))
        print(f"Chosen test item: {test_item}")

        # Recommend
        recs = recommend_recipes_train(random_user, k=10)
        print(f"Top-10 Recs: {recs}")
        if test_item in recs:
            print("✅ Test item is recommended.")
        else:
            print("❌ Test item NOT recommended. This explains zero hit.")

    # Run the debug routine on one random user
    debug_evaluation_issues()
    ##############################################################################
    # 10. Extended evaluation for all testable users (K=1..10)
    ##############################################################################
    print("\n--- EXTENDED EVALUATION ON ALL TESTABLE USERS (K=1..10) ---")

    k_values = range(1, 11)  # Evaluate K=1..10

    # We'll re-use the existing metric functions you defined above:
    # - precision_at_k
    # - hit_rate_at_k
    # - ndcg_at_k
    # - map_at_k

    # We'll track the sum of each metric over all testable users for each K
    metrics_sums = {
        'precision': np.zeros(len(k_values), dtype=np.float32),
        'hit_rate':  np.zeros(len(k_values), dtype=np.float32),
        'ndcg':      np.zeros(len(k_values), dtype=np.float32),
        'map':       np.zeros(len(k_values), dtype=np.float32),
    }

    testable_user_count = 0

    # Evaluate on every user in [0..num_users-1], where `num_users`
    # is the shape of the sliced train_set_sparse (common_users).
    for user_id in range(num_users):
        # Attempt to get a valid LOO test item
        test_item = leave_one_out_cv_test(user_id)
        if test_item is None:
            # This user has no test item that appears in training => skip
            continue

        # We only have one ground-truth item for LOO
        ground_truth = [test_item]
        # Retrieve up to top-10 recommended items
        recs_top_10 = recommend_recipes_train(user_id, k=10)
        
        testable_user_count += 1  # We found a valid test item => "testable" user

        # For each K in 1..10, compute metrics
        for idx, k_ in enumerate(k_values):
            p  = precision_at_k(recs_top_10, ground_truth, k_)
            hr = hit_rate_at_k(recs_top_10, ground_truth, k_)
            nd = ndcg_at_k(recs_top_10, ground_truth, k_)
            ma = map_at_k(recs_top_10, ground_truth, k_)

            metrics_sums['precision'][idx] += p
            metrics_sums['hit_rate'][idx]  += hr
            metrics_sums['ndcg'][idx]      += nd
            metrics_sums['map'][idx]       += ma

    # Only print results if at least one user was testable
    if testable_user_count > 0:
        print(f"\nEvaluated on {testable_user_count} testable users (LOO). Results for K=1..10:")
        # Compute average (divide each metric sum by # of testable users)
        avg_precision = metrics_sums['precision'] / testable_user_count
        avg_hitrate   = metrics_sums['hit_rate']   / testable_user_count
        avg_ndcg      = metrics_sums['ndcg']       / testable_user_count
        avg_map       = metrics_sums['map']        / testable_user_count

        for idx, k_ in enumerate(k_values):
            print(f"K={k_:2d} | "
                f"Precision={avg_precision[idx]:.4f} | "
                f"HitRate={avg_hitrate[idx]:.4f} | "
                f"NDCG={avg_ndcg[idx]:.4f} | "
                f"MAP={avg_map[idx]:.4f}")
    else:
        print("No testable users found in the entire dataset. Extended evaluation skipped.")


if __name__ == "__main__":
    main()