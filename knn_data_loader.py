import torch
import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def load_split_data_80_10_10(dataset_folder="data/"):
    """
    Loads user–recipe interactions, creates a sparse matrix,
    and splits the interactions into train, validation, and test (80/10/10)
    on a per-user basis.
    
    Returns:
        train_set (csr_matrix), val_set (csr_matrix), test_set (csr_matrix)
    """

    # 1. Load the user–recipe edge data
    #    This file is expected to contain something like:
    #      [ (all_u2r_src, all_u2r_dst, all_u2r_weight) ]
    #    in a list, which we then unpack.
    all_u2r_src_dst_weight = torch.load(
        os.path.join(dataset_folder, "all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt")
    )
    all_u2r_src, all_u2r_dst, all_u2r_weight = all_u2r_src_dst_weight[0]

    # 2. Convert lists (if they are Python lists) to PyTorch tensors
    all_u2r_src = torch.tensor(all_u2r_src, dtype=torch.long)
    all_u2r_dst = torch.tensor(all_u2r_dst, dtype=torch.long)
    all_u2r_weight = torch.tensor(all_u2r_weight, dtype=torch.float)

    # 3. Define the total number of users and recipes (known from the dataset)
    num_users = 7959
    num_recipes = 68794

    # 4. Create a sparse User–Recipe interaction matrix of shape (num_users x num_recipes)
    interaction_matrix = csr_matrix(
        (all_u2r_weight.numpy(), (all_u2r_src.numpy(), all_u2r_dst.numpy())),
        shape=(num_users, num_recipes),
    )

    # 5. Prepare empty CSR matrices for train, val, test of the same shape
    #    We'll move user-specific interactions into the appropriate matrix.
    train_set = csr_matrix(interaction_matrix.shape, dtype=np.float32)
    val_set   = csr_matrix(interaction_matrix.shape, dtype=np.float32)
    test_set  = csr_matrix(interaction_matrix.shape, dtype=np.float32)

    # 6. Split interactions per user
    for user_id in range(num_users):
        # (a) Find all recipes that this user interacted with
        user_row = interaction_matrix[user_id]
        
        # user_row.indices = the list of recipe indices the user has non-zero interactions with
        # user_row.data    = the actual rating/weight for each interaction
        recipe_indices = user_row.indices
        recipe_weights = user_row.data
        
        if len(recipe_indices) == 0:
            continue  # This user has no interactions
        
        # (b) Shuffle the recipe indices (and keep track of corresponding weights)
        perm = np.random.permutation(len(recipe_indices))
        recipe_indices = recipe_indices[perm]
        recipe_weights = recipe_weights[perm]

        # (c) Compute how many go to train, val, test
        n_total = len(recipe_indices)
        n_train = int(0.8 * n_total)
        n_val   = int(0.1 * n_total)
        # Remainder goes to test
        n_test  = n_total - n_train - n_val

        # (d) Slice out the interactions
        train_recipes = recipe_indices[:n_train]
        train_values  = recipe_weights[:n_train]

        val_recipes = recipe_indices[n_train : n_train + n_val]
        val_values  = recipe_weights[n_train : n_train + n_val]

        test_recipes = recipe_indices[n_train + n_val:]
        test_values  = recipe_weights[n_train + n_val:]

        # (e) Assign these interactions to the respective sparse matrices
        if len(train_recipes) > 0:
            train_set[user_id, train_recipes] = train_values
        
        if len(val_recipes) > 0:
            val_set[user_id, val_recipes] = val_values
        
        if len(test_recipes) > 0:
            test_set[user_id, test_recipes] = test_values

    # 7. Eliminate zeros in the sparse matrices (cleans up any unused structure)
    train_set.eliminate_zeros()
    val_set.eliminate_zeros()
    test_set.eliminate_zeros()

    # 8. (Optional) Normalize the training set row-wise 
    #    so that user embeddings won't be all zeros and to keep magnitude consistent.
    #    It's common to only normalize the training split if you're constructing user embeddings from it.
    train_set = normalize(train_set, norm='l2', axis=1)
    # Convert back to sparse csr explicitly
    train_set = csr_matrix(train_set)

    # Note: You could also consider normalizing val/test in the same way,
    # but that depends on your approach. Usually, test sets are left unnormalized
    # so you can measure raw performance.

    return train_set, val_set, test_set

if __name__ == "__main__":
    train_set, val_set, test_set = load_split_data_80_10_10()
    print("Shapes:")
    print("Train:", train_set.shape, "Non-zeros:", train_set.nnz)
    print("Val:",   val_set.shape,   "Non-zeros:", val_set.nnz)
    print("Test:",  test_set.shape,  "Non-zeros:", test_set.nnz)
    print("Data loaded and split successfully.")

