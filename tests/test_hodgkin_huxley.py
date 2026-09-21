import pytest
import numpy as np
from src.models.hodgkin_huxley import HHNeuron


def test_steady_state_resting():
    """Vérifie que les valeurs au repos (-65 mV) sont dans [0, 1]."""
    neuron = HHNeuron()
    m0, h0, n0 = neuron.steady_state(-65.0)
    assert 0.0 < m0 < 0.1
    assert 0.5 < h0 < 0.7
    assert 0.2 < n0 < 0.4


def test_derivatives_equilibrium():
    """Au repos, dV/dt et les variations de portes doivent être très faibles sans courant externe."""
    neuron = HHNeuron()
    m0, h0, n0 = neuron.steady_state(-65.0)
    dV, dm, dh, dn = neuron.derivatives(t=0.0, V=-65.0, m=m0, h=h0, n=n0)
    assert abs(dm) < 1e-3
    assert abs(dh) < 1e-3
    assert abs(dn) < 1e-3
    # Sans courant injecté, dV/dt au repos est petit
    assert abs(dV) < 5.0


def test_action_potential_trigger():
    """Un fort courant dépolarisant doit induire un dV/dt positif."""
    neuron = HHNeuron(I_ext=lambda t: 20.0)
    m0, h0, n0 = neuron.steady_state(-65.0)
    dV, _, _, _ = neuron.derivatives(t=1.0, V=-65.0, m=m0, h=h0, n=n0)
    assert dV > 10.0
