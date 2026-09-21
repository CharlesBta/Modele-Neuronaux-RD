"""
Fonctions utilitaires pour la visualisation des potentiels d'action et variables synaptiques.
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_simulation(
    t: np.ndarray,
    voltages: Dict[str, np.ndarray],
    synapse_states: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Simulation de neurones Hodgkin-Huxley couplés",
) -> None:
    """
    Trace les signaux de tension des neurones et l'état synaptique optionnel.
    """
    n_plots = 1 + (1 if synapse_states is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Graphique des tensions
    ax_v = axes[0]
    for name, v in voltages.items():
        ax_v.plot(t, v, label=name)
    ax_v.set_ylabel("Potentiel V (mV)")
    ax_v.set_title(title)
    ax_v.legend(loc="upper right")
    ax_v.grid(True, linestyle="--", alpha=0.6)

    # Graphique des synapses si présent
    if synapse_states is not None:
        ax_s = axes[1]
        if synapse_states.ndim == 1:
            ax_s.plot(t, synapse_states, label="s(t)", color="green")
        else:
            for idx, s in enumerate(synapse_states):
                ax_s.plot(t, s, label=f"synapse {idx}")
        ax_s.set_ylabel("Porte synaptique s")
        ax_s.set_xlabel("Temps (ms)")
        ax_s.legend(loc="upper right")
        ax_s.grid(True, linestyle="--", alpha=0.6)
    else:
        ax_v.set_xlabel("Temps (ms)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
