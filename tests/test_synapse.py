import pytest
import numpy as np
from src.models.synapse import Synapse


def test_transmitter_release_sigmoidal():
    """La libération de neurotransmetteur doit être ~0 au repos (-65 mV) et saturée lors d'un pic (+20 mV)."""
    syn = Synapse()
    t_rest = syn.transmitter_release(-65.0)
    t_spike = syn.transmitter_release(20.0)
    assert t_rest < 1e-4
    assert t_spike > 0.95


def test_synaptic_current_direction():
    """Pour une synapse excitatrice (E_syn=0), si V_post = -65 mV et s > 0, le courant est négatif (dépolarisant)."""
    syn_exc = Synapse(g_syn=0.5, E_syn=0.0)
    I_exc = syn_exc.current(V_post=-65.0, s=0.5)
    assert I_exc < 0.0  # - I_syn dans dV/dt ajoute un terme positif (dépolarisation)

    syn_inh = Synapse(g_syn=0.5, E_syn=-80.0)
    I_inh = syn_inh.current(V_post=-65.0, s=0.5)
    assert I_inh > 0.0  # Hyperpolarisant
