import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Computes NT-Xent loss across paired feature vectors (query vs. match) within the same image.
    """

    def __init__(self, temp: float = 0.1):
        super().__init__()
        self.temp = temp

    def forward(self, query: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, N, D] tensor of query feature vectors
            match: [B, N, D] tensor of corresponding positive feature vectors
        Returns:
            Scalar loss (averaged NT-Xent across batch)
        """
        assert query.shape == match.shape and query.ndim == 3

        batch_size, num_samples, feat_dim = query.shape
        total = 0.0

        identity_template = torch.eye(num_samples, dtype=torch.bool, device=query.device)

        for b in range(batch_size):
            q_vecs = F.normalize(query[b], dim=1)
            m_vecs = F.normalize(match[b], dim=1)
            combined = torch.cat([q_vecs, m_vecs], dim=0)  # [2N, D]

            sim = torch.matmul(combined, combined.T) / self.temp
            sim_exp = sim.exp()  # [2N, 2N]

            # Construct pairwise match mask (positives)
            m_pos = torch.zeros_like(sim_exp, dtype=torch.bool)
            m_pos[:num_samples, num_samples:] = identity_template
            m_pos[num_samples:, :num_samples] = identity_template

            # All off-diagonal pairs are valid negatives
            m_neg = ~torch.eye(2 * num_samples, dtype=torch.bool, device=query.device)

            pos_vals = sim_exp[m_pos].view(2 * num_samples, 1)
            neg_vals = sim_exp[m_neg].view(2 * num_samples, -1).sum(dim=1, keepdim=True)

            batch_loss = -torch.log(pos_vals / (pos_vals + neg_vals)).sum()
            total += batch_loss

        return total / (2 * num_samples * batch_size)
