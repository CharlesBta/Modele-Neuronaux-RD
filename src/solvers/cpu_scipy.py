"""
Solveur de référence CPU basé sur scipy.integrate.solve_ivp.
Intègre les équations différentielles ordinaires couplées de Hodgkin-Huxley et synapses cinétiques.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from src.models.hodgkin_huxley import HHNeuron
from src.models.synapse import Synapse
from src.topologies.connectome import Connectome


class CPUScipyNetwork:
    """
    Assemble les neurones HHNeuron et les synapses Synapse en un système ODE global
    résolu avec scipy.integrate.solve_ivp.
    """

    def __init__(self, neurons: List[HHNeuron], synapses: List[Synapse]):
        self.neurons = neurons
        self.synapses = synapses
        self.n_neurons = len(neurons)
        self.n_synapses = len(synapses)

    @classmethod
    def from_connectome(
        cls,
        neurons: List[HHNeuron],
        connectome: Connectome,
        synapse_factory: Any = None,
        **default_synapse_kwargs,
    ) -> "CPUScipyNetwork":
        """
        Instancie le réseau à partir d'une liste de neurones et d'une topologie Connectome.
        """
        if len(neurons) != connectome.n_neurons:
            raise ValueError(
                f"Incohérence : {len(neurons)} neurones fournis pour un connectome de taille {connectome.n_neurons}"
            )

        synapses: List[Synapse] = []
        for pre_idx, post_idx, g_syn in connectome.get_edges():
            kwargs = {**default_synapse_kwargs, "g_syn": g_syn}
            if synapse_factory is not None:
                syn = synapse_factory(pre=neurons[pre_idx], post=neurons[post_idx], **kwargs)
            else:
                syn = Synapse(pre=neurons[pre_idx], post=neurons[post_idx], **kwargs)
            synapses.append(syn)

        return cls(neurons=neurons, synapses=synapses)

    def initial_state(self, V0: float = -65.0) -> np.ndarray:
        """Construit le vecteur d'état initial y0."""
        y0: List[float] = []
        for neuron in self.neurons:
            m0, h0, n0 = neuron.steady_state(V0)
            y0.extend([V0, m0, h0, n0])
        y0.extend([0.0] * self.n_synapses)
        return np.array(y0, dtype=float)

    def unpack(self, y: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Découpe le vecteur d'état plat y en états individuels."""
        neuron_states = [y[4 * i : 4 * i + 4] for i in range(self.n_neurons)]
        syn_states = y[4 * self.n_neurons :]
        return neuron_states, syn_states

    def rhs(self, t: float, y: np.ndarray) -> List[float]:
        """Fonction dérivée dy/dt pour solve_ivp."""
        neuron_states, syn_states = self.unpack(y)
        V = [state[0] for state in neuron_states]

        # Courants synaptiques cumulés par neurone postsynaptique
        I_syn_per_neuron = [0.0] * self.n_neurons
        ds_list: List[float] = []
        for syn, s in zip(self.synapses, syn_states):
            i_pre = self.neurons.index(syn.pre)
            i_post = self.neurons.index(syn.post)
            I_syn_per_neuron[i_post] += syn.current(V[i_post], s)
            ds_list.append(syn.ds_dt(V[i_pre], s))

        # Dérivées de membrane pour chaque neurone
        dydt: List[float] = []
        for i, (neuron, state) in enumerate(zip(self.neurons, neuron_states)):
            Vi, mi, hi, ni = state
            dV, dm, dh, dn = neuron.derivatives(t, Vi, mi, hi, ni, I_syn=I_syn_per_neuron[i])
            dydt.extend([dV, dm, dh, dn])

        dydt.extend(ds_list)
        return dydt

    def simulate(self, t_span: Tuple[float, float], V0: float = -65.0, **solve_ivp_kwargs) -> Any:
        """Intègre le système sur l'intervalle temporel t_span."""
        y0 = self.initial_state(V0)
        return solve_ivp(self.rhs, t_span, y0, **solve_ivp_kwargs)

    def voltages(self, sol: Any) -> Dict[str, np.ndarray]:
        """Extrait l'évolution temporelle de la tension de chaque neurone."""
        return {neuron.name: sol.y[4 * i] for i, neuron in enumerate(self.neurons)}
