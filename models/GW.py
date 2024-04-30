import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch

def IPOT_distance_torch_batch(C, beta=0.5, epsilon=1e-8):
    """
    Compute the IPOT distances for a batch of cost matrices C and regularization beta.
    This function returns a tensor of distances, one for each batch element.
    """
    batch_size, n, m = C.size()
    T = torch.ones((batch_size, n, m), dtype=C.dtype, device=C.device, requires_grad=False) / m
    sigma = torch.ones((batch_size, m, 1), dtype=C.dtype, device=C.device, requires_grad=False) / m

    A = torch.exp(-C / beta)
    A = torch.clamp(A, min=epsilon)  # Avoid values too small

    for _ in range(50):  # Number of iterations
        Q = A * T
        for _ in range(1):  # Inner loop
            delta = 1 / (torch.bmm(Q, sigma) * n + epsilon)
            sigma = 1 / (torch.bmm(Q.transpose(1, 2), delta) * m + epsilon)
            T = torch.bmm(torch.diag_embed(delta.squeeze(-1)), Q)
            T = torch.bmm(T, torch.diag_embed(sigma.squeeze(-1)))

    # Compute the IPOT distances for each batch
    distances = torch.stack([torch.trace(torch.matmul(C[b], T[b].t())) for b in range(batch_size)])

    return distances

def normalize_and_convert_to_cost_matrix(A):
    """Normalize the adjacency matrix and convert to cost matrix."""
    row_sums = A.sum(dim=2, keepdim=True) + 1e-10  # Avoid division by zero
    A_normalized = A / row_sums
    return 1 - A_normalized  # Convert to cost matrix

def IPOT_batch(C, beta=0.5, iteration=50):
    """IPOT for batch processing of multiple cost matrices."""
    bs, n, m = C.size()
    sigma = torch.ones(bs, m, 1, device=C.device) / m
    T = torch.ones(bs, n, m, device=C.device)
    A = torch.exp(-C / beta).float()
    for _ in range(iteration):
        Q = A * T
        delta = 1 / (n * torch.bmm(Q, sigma))
        sigma = 1 / (m * torch.bmm(Q.transpose(1, 2), delta))
        T = delta * Q * sigma.transpose(2, 1)
    return T

def compute_gw_distances(A, lamda=1e-1, iteration=5, OT_iteration=20):
    """Compute GW distances using fully vectorized operations."""
    C = normalize_and_convert_to_cost_matrix(A)
    bs = C.size(0)
    expanded_C = C.unsqueeze(1).expand(bs, bs, -1, -1)
    transposed_C = expanded_C.transpose(2, 3)
    gamma = IPOT_batch(expanded_C.reshape(bs * bs, *C.shape[1:]), beta=lamda, iteration=OT_iteration)
    gamma_rev = IPOT_batch(transposed_C.reshape(bs * bs, *C.shape[1:]), beta=lamda, iteration=OT_iteration)
    gamma = gamma.reshape(bs, bs, *C.shape[1:])
    gamma_rev = gamma_rev.reshape(bs, bs, *C.shape[1:])
    result = torch.sum(expanded_C * gamma, dim=(2, 3)) + torch.sum(transposed_C * gamma_rev, dim=(2, 3))
    return result.diag()  # Assuming diagonal contains the self-distances


# # # Calculate cosine cost matrix
# cosine_cost = 1 - torch.matmul(h.reshape((full_shape[0],full_shape[1],-1)), h.reshape((full_shape[0],full_shape[1],-1)).transpose(1,2))

# # Prune with threshold
# _beta = 0.2
# minval = torch.min(cosine_cost)
# maxval = torch.max(cosine_cost)
# threshold = minval + _beta * (maxval - minval)
# cosine_cost = torch.nn.functional.relu(cosine_cost - threshold)

# # Calculate OT loss using IPOT distance function
# wd = IPOT_distance_torch_batch(cosine_cost)
