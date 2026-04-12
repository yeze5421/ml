"""Neighbor graph construction utilities."""
from __future__ import annotations

import torch


def build_radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build directed radius graph.

    Returns edge_index (2,E), edge_vec (E,3), edge_dist (E,).
    """
    device = pos.device
    edge_src = []
    edge_dst = []
    edge_vec = []
    edge_dist = []

    for mol_id in torch.unique(batch):
        idx = torch.where(batch == mol_id)[0]
        p = pos[idx]
        dmat = torch.cdist(p, p)
        n = p.shape[0]
        for i in range(n):
            neighbors = torch.where((dmat[i] < cutoff) & (dmat[i] > 0))[0]
            if neighbors.numel() > max_neighbors:
                neighbor_dist = dmat[i, neighbors]
                _, order = torch.topk(neighbor_dist, k=max_neighbors, largest=False)
                neighbors = neighbors[order]
            for j in neighbors.tolist():
                src = idx[i].item()
                dst = idx[j].item()
                vec = pos[dst] - pos[src]
                edge_src.append(src)
                edge_dst.append(dst)
                edge_vec.append(vec)
                edge_dist.append(torch.norm(vec))

    if not edge_src:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        evec = torch.empty((0, 3), dtype=pos.dtype, device=device)
        edist = torch.empty((0,), dtype=pos.dtype, device=device)
        return edge_index, evec, edist

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    evec = torch.stack(edge_vec).to(device)
    edist = torch.stack(edge_dist).to(device)
    return edge_index, evec, edist
