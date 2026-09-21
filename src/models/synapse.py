"""
Modèle biophysique de synapse chimique à conductance.
Cinétique du premier ordre (Destexhe, Mainen & Sejnowski, 1994).
"""

from typing import Any, Optional
import numpy as np


class Synapse:
    """
    Modèle de synapse chimique avec dynamique de liaison de neurotransmetteurs.

    Équation différentielle de porte synaptique :
        ds/dt = alpha * T(V_pre) * (1 - s) - beta * s

    Courant postsynaptique injecté :
        I_syn = g_syn * s * (V_post - E_syn)
    """

    def __init__(
        self,
        g_syn: float = 0.5,
        E_syn: float = 0.0,
        alpha: float = 1.1,
        beta: float = 0.19,
        T_max: float = 1.0,
        V_T: float = 2.0,
        K_p: float = 5.0,
        pre: Optional[Any] = None,
        post: Optional[Any] = None,
    ):
        """
        Paramètres
        ----------
        g_syn : float
            Conductance synaptique maximale (mS/cm^2).
        E_syn : float
            Potentiel d'inversion (mV). 0 mV (excitateur AMPA), -80 mV (inhibiteur GABA_A).
        alpha : float
            Taux de liaison (1/(mM*ms)).
        beta : float
            Taux de dissociation (1/ms).
        T_max : float
            Concentration maximale de neurotransmetteur (mM).
        V_T : float
            Tension de demi-activation de libération (mV).
        K_p : float
            Pente de la sigmoïde de libération (mV).
        pre, post : Any, optional
            Références vers les neurones pré et postsynaptiques.
        """
        self.g_syn = g_syn
        self.E_syn = E_syn
        self.alpha = alpha
        self.beta = beta
        self.T_max = T_max
        self.V_T = V_T
        self.K_p = K_p
        self.pre = pre
        self.post = post

    def transmitter_release(self, V_pre: float) -> float:
        """
        Concentration de neurotransmetteur libéré en fonction du potentiel présynaptique.
        """
        return float(self.T_max / (1.0 + np.exp(-(V_pre - self.V_T) / self.K_p)))

    def current(self, V_post: float, s: float) -> float:
        """
        Courant synaptique injecté dans le compartiment postsynaptique.
        """
        return float(self.g_syn * s * (V_post - self.E_syn))

    def ds_dt(self, V_pre: float, s: float) -> float:
        """
        Dérivée temporelle de la fraction de récepteurs liés s.
        """
        T_rel = self.transmitter_release(V_pre)
        return float(self.alpha * T_rel * (1.0 - s) - self.beta * s)
