"""
Modèle biophysique de neurone de Hodgkin-Huxley (1952).
Axone géant de calmar avec canaux ioniques Na+, K+ et courant de fuite.
"""

from typing import Callable, Optional, Tuple
import numpy as np


class HHNeuron:
    """
    Représente un compartiment membranaire selon le modèle de Hodgkin-Huxley.

    Variables d'état :
        V : Potentiel de membrane (mV)
        m : Activation du sodium (adimensionnel, [0, 1])
        h : Inactivation du sodium (adimensionnel, [0, 1])
        n : Activation du potassium (adimensionnel, [0, 1])
    """

    def __init__(
        self,
        C_m: float = 1.0,
        g_Na: float = 120.0,
        g_K: float = 36.0,
        g_L: float = 0.3,
        E_Na: float = 50.0,
        E_K: float = -77.0,
        E_L: float = -54.387,
        I_ext: Optional[Callable[[float], float]] = None,
        name: str = "neuron",
    ):
        """
        Initialise les paramètres biophysiques du neurone.

        Parameters
        ----------
        C_m : float
            Capacité membranaire (uF/cm^2).
        g_Na, g_K, g_L : float
            Conductances maximales respectives (mS/cm^2).
        E_Na, E_K, E_L : float
            Potentiels d'inversion Nernst respectifs (mV).
        I_ext : Callable[[float], float], optional
            Fonction du temps injectant un courant externe (uA/cm^2).
        name : str
            Identifiant textuel du neurone.
        """
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.I_ext = I_ext if I_ext is not None else (lambda t: 0.0)
        self.name = name

    @staticmethod
    def alpha_m(V: float) -> float:
        """Taux d'ouverture de la porte d'activation du Na+ (rapide)."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    @staticmethod
    def beta_m(V: float) -> float:
        """Taux de fermeture de la porte d'activation du Na+."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V: float) -> float:
        """Taux de désinactivation de la porte h du Na+."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def beta_h(V: float) -> float:
        """Taux d'inactivation de la porte h du Na+."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def alpha_n(V: float) -> float:
        """Taux d'ouverture de la porte d'activation du K+ (lente)."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    @staticmethod
    def beta_n(V: float) -> float:
        """Taux de fermeture de la porte d'activation du K+."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def steady_state(self, V: float) -> Tuple[float, float, float]:
        """
        Calcule les valeurs d'équilibre asymptotiques (m_inf, h_inf, n_inf) pour une tension V.
        """
        m0 = self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))
        h0 = self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))
        n0 = self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))
        return float(m0), float(h0), float(n0)

    def derivatives(
        self, t: float, V: float, m: float, h: float, n: float, I_syn: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """
        Calcule les dérivées temporelles (dV/dt, dm/dt, dh/dt, dn/dt).
        """
        I_Na = self.g_Na * (m**3) * h * (V - self.E_Na)
        I_K = self.g_K * (n**4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Bilan de courant sur la membrane
        dV = (self.I_ext(t) - I_Na - I_K - I_L - I_syn) / self.C_m

        # Évolution des variables de porte
        dm = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dh = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dn = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n

        return float(dV), float(dm), float(dh), float(dn)
