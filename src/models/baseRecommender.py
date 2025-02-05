from abc import ABC, abstractmethod
import time
import torch

from src.metrics import get_performance_all_users

class Recommender(ABC):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @abstractmethod
    def fit(self, data):
        pass

    def evaluate(self, dataloader, topk=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], verbose=True, return_per_user=False):
        if self.model is None:
            raise ValueError("No model has been set in this Recommender.")

        self.model.eval()
        user2pos_score_dict = {}
        user2neg_score_dict = {}
        if verbose:
            start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                edge_score = self.model(batch)  # shape [num_edges]
                labels = batch['user', 'u-r', 'recipe'].edge_label  # shape [num_edges]
                num_pos = int((labels == 1).sum().item())

                pos_score = edge_score[:num_pos]
                neg_score = edge_score[num_pos:]

                # get user indices
                user_edge_index = batch['user', 'u-r', 'recipe'].edge_index[0].cpu().tolist()
                pos_scores_list = pos_score.cpu().tolist()
                neg_scores_list = neg_score.cpu().tolist()

                # accumulate scores in dictionary
                for i in range(num_pos):
                    user = user_edge_index[i]
                    user2pos_score_dict.setdefault(user, []).append(pos_scores_list[i])
                for i in range(num_pos, len(user_edge_index)):
                    user = user_edge_index[i]
                    neg_index = i - num_pos  # shift the index so it matches the 0-based index in neg_scores_list
                    user2neg_score_dict.setdefault(user, []).append(neg_scores_list[neg_index])

        evaluation_result = get_performance_all_users(user2pos_score_dict, user2neg_score_dict, topk)

        if verbose:
            elapsed = time.strftime("%M:%S min", time.gmtime(time.time() - start_time))
            print(f"Evaluation done in {elapsed}.")
            print("Evaluation metrics:", evaluation_result)

        if return_per_user:
            return evaluation_result, user2pos_score_dict, user2neg_score_dict
        return evaluation_result