import pytest
import numpy as np
from src.topologies.connectome import Connectome


def test_connectome_chain():
    conn = Connectome.make_chain(n_neurons=4, g_syn=0.5, periodic=False)
    assert conn.n_neurons == 4
    edges = conn.get_edges()
    assert len(edges) == 3
    assert (0, 1, 0.5) in edges
    assert (1, 2, 0.5) in edges
    assert (2, 3, 0.5) in edges


def test_connectome_all_to_all():
    conn = Connectome.make_all_to_all(n_neurons=5, g_syn=0.2, self_loops=False)
    edges = conn.get_edges()
    # 5 * 4 = 20 arêtes
    assert len(edges) == 20
    assert np.all(np.diag(conn.weights) == 0.0)
