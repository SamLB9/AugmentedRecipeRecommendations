import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from sklearn.manifold import TSNE
import faiss

def load_split_data(dataset_folder = "data/"):
    # 1. Load edge data (Recipe–Ingredient)
    r_i_src, r_i_dst, r_i_weight = torch.load(
        os.path.join(dataset_folder, "edge_r2i_src_dst_weight.pt")
    )

    # 2. Load edge data (Recipe–Recipe)
    r_r_src, r_r_dst, r_r_weight = torch.load(
        os.path.join(dataset_folder, "edge_r2r_src_and_dst_and_weight.pt")
    )

    # 3. Load edge data (Ingredient–Ingredient)
    i_i_src, i_i_dst, i_i_weight = torch.load(
        os.path.join(dataset_folder, "edge_i2i_src_and_dst_and_weight.pt")
    )

    # 4. Load edge data (User–Recipe)
    all_u2r_src_dst_weight = torch.load(
        os.path.join(dataset_folder, "all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt")
    )
    all_u2r_src, all_u2r_dst, all_u2r_weight = all_u2r_src_dst_weight[0]

    # 5. Load node data (Recipe/Ingredient)
    recipe_instr_features = torch.load(
        os.path.join(dataset_folder, "recipe_nodes_avg_instruction_features.pt")
    )
    ingredient_nutrient_features = torch.load(
        os.path.join(dataset_folder, "ingredient_nodes_nutrient_features.pt")
    )

    data = HeteroData()

    # Convert lists to tensors if necessary
    all_u2r_src = torch.tensor(all_u2r_src, dtype=torch.long)
    all_u2r_dst = torch.tensor(all_u2r_dst, dtype=torch.long)
    all_u2r_weight = torch.tensor(all_u2r_weight, dtype=torch.float)

    r_i_src = torch.tensor(r_i_src, dtype=torch.long)
    r_i_dst = torch.tensor(r_i_dst, dtype=torch.long)
    r_i_weight = torch.tensor(r_i_weight, dtype=torch.float)

    r_r_src = torch.tensor(r_r_src, dtype=torch.long)
    r_r_dst = torch.tensor(r_r_dst, dtype=torch.long)
    r_r_weight = torch.tensor(r_r_weight, dtype=torch.float)

    i_i_src = torch.tensor(i_i_src, dtype=torch.long)
    i_i_dst = torch.tensor(i_i_dst, dtype=torch.long)
    i_i_weight = torch.tensor(i_i_weight, dtype=torch.float)

    # NODES:
    # We often need to define the number of nodes per type.
    # For example, from your logs:
    num_users = 7959
    num_recipes = 68794
    num_ingredients = 8847

    data["user"].num_nodes = num_users
    data["recipe"].num_nodes = num_recipes
    data["ingredient"].num_nodes = num_ingredients

    # EDGES:
    # user -> recipe
    data["user", "u-r", "recipe"].edge_index = torch.stack([all_u2r_src, all_u2r_dst], dim=0)
    data["user", "u-r", "recipe"].edge_weight = all_u2r_weight

    # recipe -> ingredient
    data["recipe", "r-i", "ingredient"].edge_index = torch.stack([r_i_src, r_i_dst], dim=0)
    data["recipe", "r-i", "ingredient"].edge_weight = r_i_weight

    # recipe -> recipe
    data["recipe", "r-r", "recipe"].edge_index = torch.stack([r_r_src, r_r_dst], dim=0)
    data["recipe", "r-r", "recipe"].edge_weight = r_r_weight

    # ingredient -> ingredient
    data["ingredient", "i-i", "ingredient"].edge_index = torch.stack([i_i_src, i_i_dst], dim=0)
    data["ingredient", "i-i", "ingredient"].edge_weight = i_i_weight

    # If you want to make the graph fully bidirectional:
    data = ToUndirected()(data)

    # recipe_instr_features: [68794, feature_dim_recipe]
    # ingredient_nutrient_features: [8847, feature_dim_ingr]
    data["recipe"].x = recipe_instr_features
    data["ingredient"].x = ingredient_nutrient_features

    # Suppose we use random user features, e.g., shape [7959, 300]:
    #user_feat = torch.rand(num_users, 300)
    #data["user"].x = user_feat

    # Load the HeteroData graph (assumed to be preloaded as 'data')
    num_users = data["user"].num_nodes
    num_recipes = data["recipe"].num_nodes

    # Extract user-recipe interaction data
    user_recipe_src = data["user", "u-r", "recipe"].edge_index[0]
    user_recipe_dst = data["user", "u-r", "recipe"].edge_index[1]
    user_recipe_weight = data["user", "u-r", "recipe"].edge_weight

    # Create a sparse user-recipe interaction matrix
    interaction_matrix = csr_matrix((user_recipe_weight.numpy(), (user_recipe_src.numpy(), user_recipe_dst.numpy())),
                                    shape=(num_users, num_recipes))

    # Normalize interaction matrix
    interaction_matrix = normalize(interaction_matrix, norm='l2', axis=1)

    # Convert to dense numpy array (only if memory allows)
    user_embeddings = interaction_matrix.toarray().astype(np.float32)

    # Now, split user_embeddings into 8:1:1 (train:validation:test)
    n = user_embeddings.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)
    
    train_set = user_embeddings[indices[:train_end]]
    valid_set = user_embeddings[indices[train_end:valid_end]]
    test_set  = user_embeddings[indices[valid_end:]]

    return train_set, valid_set, test_set

if __name__ == "__main__":
    train_set, valid_set, test_set = load_split_data()
    print("Data loaded successfully.")