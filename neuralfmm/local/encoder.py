import torch
import torch.nn as nn

from ..utils.cutoffs import bessel_rbf, cosine_cutoff
from .neighbors import edge_vectors, periodic_neighbor_list


class PaiNNMessage(nn.Module):
    """Equivariant message block (Schutt et al., 2021).

    Scalars s (N, F) are invariant; vectors v (N, 3, F) transform as vectors
    under O(3) because they are only ever built from linear combinations of
    other vectors and elementwise gates that are themselves invariant scalars.
    """

    def __init__(self, hidden_dim, n_rbf, r_cut):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_cut = r_cut
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        self.filter_net = nn.Linear(n_rbf, 3 * hidden_dim)

    def forward(self, s, v, edge_index, vectors):
        src, dst = edge_index
        dist = torch.sqrt((vectors**2).sum(dim=-1).clamp_min(1e-12))
        dir_unit = vectors / dist.clamp_min(1e-8).unsqueeze(-1)

        rbf = bessel_rbf(dist, self.r_cut, self.filter_net.in_features)
        envelope = cosine_cutoff(dist, self.r_cut).unsqueeze(-1)
        w = self.filter_net(rbf) * envelope  # (E, 3F)
        w1, w2, w3 = torch.split(w, self.hidden_dim, dim=-1)

        phi = self.phi(s[src])  # (E, 3F)
        phi1, phi2, phi3 = torch.split(phi, self.hidden_dim, dim=-1)

        ds_edge = phi1 * w1  # (E, F)
        dv_edge = v[src] * (phi2 * w2).unsqueeze(1) + (phi3 * w3).unsqueeze(1) * dir_unit.unsqueeze(-1)

        n = s.shape[0]
        ds = torch.zeros_like(s).index_add_(0, dst, ds_edge)
        dv = torch.zeros_like(v).index_add_(0, dst, dv_edge)
        return s + ds, v + dv


class PaiNNUpdate(nn.Module):
    """Intra-atomic mixing block: mixes s and v while preserving equivariance."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s, v):
        # v: (N, 3, F) -> apply linear map along the channel dim only
        Uv = torch.einsum("nkf,fg->nkg", v, self.U.weight.t())
        Vv = torch.einsum("nkf,fg->nkg", v, self.V.weight.t())

        # torch.norm's second derivative blows up (NaN) at/near Vv=0, which
        # happens often enough (e.g. cancelling channels) to break the
        # double-backward force training needs; a clamped sqrt keeps both
        # derivatives bounded everywhere.
        Vv_norm = torch.sqrt((Vv**2).sum(dim=1).clamp_min(1e-12))  # (N, F)
        stack = torch.cat([s, Vv_norm], dim=-1)
        a = self.mlp(stack)
        a_ss, a_sv, a_vv = torch.split(a, self.hidden_dim, dim=-1)

        dot = (Uv * Vv).sum(dim=1)  # (N, F)
        ds = a_ss + a_vv * dot
        dv = a_sv.unsqueeze(1) * Uv
        return s + ds, v + dv


class EquivariantEncoder(nn.Module):
    """The shared "LOCAL E(3)-EQUIVARIANT ENCODER" box: a PaiNN backbone
    (Schutt et al., 2021) producing, per atom, an invariant scalar
    descriptor `s` and an equivariant vector `v`. Both `les.LESModel` and
    `fmm.NeuralFMM` instantiate this same class and read off `s` -- LES
    feeds it to a local energy head plus a latent-charge head, NeuralFMM
    feeds it to a local energy head plus straight on as the atomic features
    `h_i` handed to the octree. Nothing here is architecture-specific.
    """

    def __init__(self, num_species, hidden_dim=64, n_layers=3, n_rbf=16, r_cut=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_cut = r_cut
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.messages = nn.ModuleList(
            [PaiNNMessage(hidden_dim, n_rbf, r_cut) for _ in range(n_layers)]
        )
        self.updates = nn.ModuleList([PaiNNUpdate(hidden_dim) for _ in range(n_layers)])

    def forward(self, positions, species, cell, graph=None):
        if graph is None:
            edge_index, shifts, _ = periodic_neighbor_list(positions, cell, self.r_cut)
        else:
            edge_index, shifts = graph
        vectors = edge_vectors(positions, cell, edge_index, shifts)
        return self._run(species, edge_index, vectors)

    def forward_batched(self, positions_list, species, cell_list, graphs=None):
        """Same computation as `forward`, but over N structures concatenated
        into one block-diagonal graph instead of looped one at a time.

        positions_list / cell_list: length-N per-structure tensors (each
        structure has its own periodic cell, so its neighbor list has to be
        built separately -- this loop only builds index tensors, it does no
        model math). species: (sum(n_i),) already concatenated in the same
        structure order as positions_list.

        graphs: optional length-N list of cached (edge_index, shifts) pairs
        (see AtomicSystem.get_neighbor_graph) -- strongly recommended for
        training, since `periodic_neighbor_list` does data-dependent
        boolean-mask indexing that forces a GPU synchronize every call, and
        this loop would otherwise pay that cost once per structure, every
        forward pass, every epoch, even though the topology never changes
        for a fixed training sample. When cached, only the differentiable
        `edge_vectors` gather (no synchronize) runs here.

        Every edge stays within the structure it came from (src/dst are
        offset per structure, never cross-linked), so the shared
        message/update stack below -- which only does index_add_ scatter
        over `edge_index`/embedding lookups, no structure-count assumption
        -- gives bit-identical per-atom features to calling `forward` once
        per structure, just in a single batched pass instead of N sequential
        ones.
        """
        edge_chunks, vector_chunks = [], []
        offset = 0
        for i, (positions, cell) in enumerate(zip(positions_list, cell_list)):
            if graphs is None:
                edge_index, shifts, _ = periodic_neighbor_list(positions, cell, self.r_cut)
            else:
                edge_index, shifts = graphs[i]
            vectors = edge_vectors(positions, cell, edge_index, shifts)
            edge_chunks.append(edge_index + offset)
            vector_chunks.append(vectors)
            offset += positions.shape[0]

        device = species.device
        edge_index = torch.cat(edge_chunks, dim=1) if edge_chunks else torch.zeros(2, 0, dtype=torch.long, device=device)
        vectors = torch.cat(vector_chunks, dim=0) if vector_chunks else torch.zeros(0, 3, device=device)
        return self._run(species, edge_index, vectors)

    def _run(self, species, edge_index, vectors):
        s = self.embedding(species)
        v = torch.zeros(species.shape[0], 3, self.hidden_dim, device=species.device, dtype=s.dtype)

        for msg, upd in zip(self.messages, self.updates):
            s, v = msg(s, v, edge_index, vectors)
            s, v = upd(s, v)

        return s, v
