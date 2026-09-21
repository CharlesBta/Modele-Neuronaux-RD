"""Modèles biophysiques membranaires et synaptiques."""

from .hodgkin_huxley import HHNeuron
from .synapse import Synapse

__all__ = ["HHNeuron", "Synapse"]
