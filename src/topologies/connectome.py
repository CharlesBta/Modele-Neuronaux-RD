"""
Gestion et génération de connectomes structurés pour réseaux d'oscillateurs neuronaux.
"""

from typing import List, Tuple, Optional
import numpy as np


class Connectome:
    """
    Représente la structure d'interconnexion synaptique d'un réseau neuronal.
    """

    def __init__(self, n_neurons: int, weights: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        n_neurons : int
            Nombre total de neurones dans le réseau.
        weights : np.ndarray, optional
            Matrice d'adjacence pondérée (conductances synaptiques g_syn), de taille (N, N).
            L'élément (i, j) représente la connexion du neurone i vers le neurone j.
        """
        self.n_neurons = n_neurons
        if weights is not None:
            if weights.shape != (n_neurons, n_neurons):
                raise ValueError(f"Shape attendue ({n_neurons}, {n_neurons}), reçu {weights.shape}")
            self.weights = weights.astype(float)
        else:
            self.weights = np.zeros((n_neurons, n_neurons), dtype=float)

    def add_synapse(self, pre_idx: int, post_idx: int, g_syn: float = 0.5) -> None:
        """Ajoute ou modifie une connexion dirigée de pre vers post."""
        if not (0 <= pre_idx < self.n_neurons and 0 <= post_idx < self.n_neurons):
            raise IndexError("Index de neurone hors bornes.")
        self.weights[pre_idx, post_idx] = float(g_syn)

    def get_edges(self) -> List[Tuple[int, int, float]]:
        """Retourne la liste des arêtes non nulles (pre_idx, post_idx, g_syn)."""
        edges = []
        pres, posts = np.nonzero(self.weights)
        for pre, post in zip(pres, posts):
            edges.append((int(pre), int(post), float(self.weights[pre, post])))
        return edges

    @classmethod
    def make_chain(cls, n_neurons: int, g_syn: float = 0.5, periodic: bool = False) -> "Connectome":
        """Génère un connectome en chaîne 1D (pre -> post)."""
        conn = cls(n_neurons)
        for i in range(n_neurons - 1):
            conn.add_synapse(i, i + 1, g_syn=g_syn)
        if periodic and n_neurons > 1:
            conn.add_synapse(n_neurons - 1, 0, g_syn=g_syn)
        return conn

    @classmethod
    def make_all_to_all(cls, n_neurons: int, g_syn: float = 0.1, self_loops: bool = False) -> "Connectome":
        """Génère un réseau complètement connecté."""
        W = np.full((n_neurons, n_neurons), g_syn, dtype=float)
        if not self_loops:
            np.fill_diagonal(W, 0.0)
        return cls(n_neurons, weights=W)

    @classmethod
    def make_random(
        cls, n_neurons: int, p_connect: float = 0.2, g_syn: float = 0.2, seed: Optional[int] = None
    ) -> "Connectome":
        """Génère un réseau aléatoire Erdős-Rényi dirigé."""
        rng = np.random.default_rng(seed)
        mask = rng.random((n_neurons, n_neurons)) < p_connect
        np.fill_diagonal(mask, False)
        W = mask.astype(float) * g_syn
        return cls(n_neurons, weights=W)
