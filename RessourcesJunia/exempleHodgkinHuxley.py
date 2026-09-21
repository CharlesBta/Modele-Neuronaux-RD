"""
Deux neurones de Hodgkin-Huxley couplés, connectés par une synapse réaliste à conductances

Version orientée objet

Modèle de neurone :  Modèle classique Hodgkin & Huxley (1952) de l'axone géant de calamar
Modèle synaptique :  Modèle de synapse chimique à conductance, basé sur un schéma
                     cinétique du premier ordre (Destexhe, Mainen & Sejnowski, 1994).

- Chaque neurone est un objet HHNeuron qui possède son propre état
  (V, m, h, n) et ses propres paramètres de canaux ioniques.
- La synapse est un objet Synapse séparé : elle "lit" la tension du
  neurone présynaptique et injecte un courant dans le neurone
  postsynaptique.
- La classe Network assemble plusieurs neurones et synapses dans un
  seul vecteur d'état pour pouvoir utiliser un solveur d'équations
  différentielles générique (scipy.integrate.solve_ivp).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class HHNeuron:
    """
    Représente un compartiment de membrane de type Hodgkin-Huxley (canaux Na+, K+ et fuite).

    Chaque neurone a 4 variables d'état qui évoluent dans le temps :
        V : potentiel de membrane (mV)
        m : variable d'activation du canal Na+  (0 = fermé, 1 = ouvert)
        h : variable d'inactivation du canal Na+ (0 = inactivé, 1 = disponible)
        n : variable d'activation du canal K+   (0 = fermé, 1 = ouvert)

    Les paramètres par défaut (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)
    correspondent aux valeurs originales mesurées par Hodgkin et Huxley
    sur l'axone géant de calmar (1952).
    """

    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387, I_ext=None, name="neuron"):
        """
        Paramètres
        ----------
        C_m : capacité membranaire (uF/cm^2). Détermine la "vitesse"
              avec laquelle V réagit à un courant net.
        g_Na, g_K, g_L : conductances maximales des canaux Na+, K+ et
              fuite (mS/cm^2). Plus elles sont grandes, plus le courant
              correspondant peut être important.
        E_Na, E_K, E_L : potentiels d'inversion (mV) de chaque canal,
              c'est-à-dire la tension vers laquelle chaque courant
              "pousse" la membrane.
        I_ext : fonction du temps I_ext(t) représentant un courant
              externe injecté (électrode, stimulation...). Si None,
              le neurone ne reçoit aucun courant externe.
        name : simple étiquette utile pour retrouver ce neurone dans
              les résultats de simulation.
        """
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.I_ext = I_ext if I_ext is not None else (lambda t: 0.0) # Si aucune fonction de courant n'est fournie, on utilise un courant nul en permanence (lambda t: 0.0).
        self.name = name

    # ------------------------------------------------------------------
    # Fonctions de taux (alpha/beta) : ce sont des fonctions empiriques
    # de la tension V, ajustées par Hodgkin et Huxley sur leurs données
    # expérimentales. Chaque paire (alpha_x, beta_x) décrit la vitesse
    # de transition d'une porte x entre l'état fermé et l'état ouvert :
    #
    #     fermé --alpha_x(V)--> ouvert
    #     ouvert --beta_x(V)--> fermé
    #
    # On les déclare en @staticmethod car elles ne dépendent que de V,
    # pas de l'état interne d'une instance particulière.
    # ------------------------------------------------------------------
    @staticmethod
    def alpha_m(V): 
        """Taux d'ouverture de la porte d'activation du Na+ (rapide)."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    @staticmethod
    def beta_m(V):  
        """Taux de fermeture de la porte d'activation du Na+."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V): 
        """Taux de 'désinactivation' de la porte h du Na+."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    @staticmethod
    def beta_h(V):  
        """Taux d'inactivation de la porte h du Na+."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def alpha_n(V): 
        """Taux d'ouverture de la porte d'activation du K+ (lente)."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    @staticmethod
    def beta_n(V):  
        """Taux de fermeture de la porte d'activation du K+."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def steady_state(self, V):
        """
        Calcule les valeurs d'équilibre (m0, h0, n0) des trois portes
        pour une tension V fixée.

        À l'équilibre, dx/dt = 0, donc :
            alpha_x(V) * (1 - x) = beta_x(V) * x
            => x_inf(V) = alpha_x(V) / (alpha_x(V) + beta_x(V))

        Utile pour initialiser une simulation "au repos" (V = -65 mV
        par exemple), sans transitoire artificiel au démarrage.
        """
        m0 = self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))
        h0 = self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))
        n0 = self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))
        return m0, h0, n0

    def derivatives(self, t, V, m, h, n, I_syn=0.0):
        """
        Calcule les dérivées temporelles (dV, dm, dh, dn) du neurone
        à l'instant t, étant donné son état actuel (V, m, h, n).

        Paramètres
        ----------
        t : temps courant (ms), utile pour évaluer I_ext(t).
        V, m, h, n : état courant du neurone.
        I_syn : courant synaptique reçu par ce neurone (en plus de
                I_ext). Convention : un courant positif est un courant
                "sortant" pour la membrane (comme I_Na, I_K, I_L),
                donc il est soustrait de la même façon que les autres
                courants ioniques.

        Retour
        ------
        (dV, dm, dh, dn) : dérivées à intégrer par le solveur.

        Détail biophysique
        ------------------
        - I_Na, I_K, I_L sont calculés avec la loi d'Ohm généralisée :
              I = g * (variable(s) de porte) * (V - E_reversion)
          Le courant s'annule quand V = E_reversion (pas de force
          motrice), et change de signe de part et d'autre.
        - dV/dt découle simplement de la loi des noeuds appliquée à
          la membrane : C_m * dV/dt = somme des courants entrants.
        - dm, dh, dn suivent chacun la même cinétique du premier ordre
          "fermé <-> ouvert" décrite plus haut.
        """
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na) # Courant sodium : m^3 reflète les 3 sous-unités d'activation indépendantes, h reflète l'inactivation.
        I_K  = self.g_K  * n**4     * (V - self.E_K)  # Courant potassium : n^4 reflète les 4 sous-unités d'activation.
        I_L  = self.g_L             * (V - self.E_L)  # Courant de fuite : linéaire, sans porte (toujours "ouvert").

        dV = (self.I_ext(t) - I_Na - I_K - I_L - I_syn) / self.C_m # Bilan des courants sur la membrane (loi des noeuds : C_m * dV/dt = I_ext - I_Na - I_K - I_L - I_syn) 
        # Cinétique des portes : vitesse d'ouverture * fraction fermée - vitesse de fermeture * fraction ouverte.
        dm = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dh = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dn = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        return dV, dm, dh, dn


class Synapse:
    """
    Modèle de synapse chimique à conductance, basé sur un schéma
    cinétique du premier ordre (Destexhe, Mainen & Sejnowski, 1994).

    Idée : une variable s(t) in [0, 1] représente la fraction de
    récepteurs postsynaptiques actuellement liés au neurotransmetteur.
    Elle évolue selon :

        ds/dt = alpha * T(V_pre) * (1 - s) - beta * s

    où T(V_pre) est une fonction sigmoïde qui vaut ~0 au repos et
    monte vers T_max quand le neurone présynaptique dépasse son seuil
    de décharge (~0 mV). Le courant synaptique injecté dans le neurone
    postsynaptique est alors :

        I_syn = g_syn * s * (V_post - E_syn)

    Ce modèle est plus réaliste qu'un simple "si V_pre dépasse un
    seuil, ajoute un courant constant", car il reproduit :
    - un temps de montée et un temps de décroissance physiologiques,
    - une saturation (s ne peut pas dépasser 1),
    - une dépendance au potentiel postsynaptique via (V_post - E_syn),
      exactement comme un vrai canal ionique synaptique.
    """

    def __init__(self, g_syn=0.5, E_syn=0.0, alpha=1.1, beta=0.19, T_max=1.0, V_T=2.0, K_p=5.0, pre=None, post=None):
        """
        Paramètres
        ----------
        g_syn : conductance synaptique maximale (mS/cm^2). Contrôle
                la "force" de la connexion.
        E_syn : potentiel d'inversion de la synapse (mV).
                - E_syn = 0 mV   -> synapse excitatrice (type AMPA)
                - E_syn = -80 mV -> synapse inhibitrice (type GABA_A)
        alpha : taux de liaison du neurotransmetteur au récepteur
                (1/(mM*ms)). Plus il est grand, plus s monte vite
                pendant un potentiel d'action présynaptique.
        beta  : taux de dissociation (1/ms). Détermine le temps de
                décroissance du courant synaptique après le pic
                présynaptique (tau_decay ~ 1/beta).
        T_max : concentration maximale de neurotransmetteur (mM).
        V_T   : tension à laquelle la libération de neurotransmetteur
                atteint la moitié de son maximum (mV).
        K_p   : pente de la sigmoïde de libération (mV). Plus K_p est
                petit, plus la transition est abrupte (proche d'un
                interrupteur tout-ou-rien autour de V_T).
        pre, post : objets HHNeuron présynaptique et postsynaptique.
                Simplement gardés en référence pour savoir qui envoie
                et qui reçoit le courant synaptique.
        """
        self.g_syn = g_syn
        self.E_syn = E_syn      # 0 mV -> excitatory, -80 mV -> inhibitory
        self.alpha = alpha
        self.beta = beta
        self.T_max = T_max
        self.V_T = V_T
        self.K_p = K_p
        self.pre = pre          # presynaptic HHNeuron (for labeling/reference)
        self.post = post        # postsynaptic HHNeuron

    def transmitter_release(self, V_pre):
        """
        Fraction de neurotransmetteur libéré en fonction de la tension
        présynaptique V_pre, sous forme d'une sigmoïde :

            T(V_pre) = T_max / (1 + exp(-(V_pre - V_T) / K_p))

        - Proche de 0 quand V_pre est au repos (pas de libération).
        - Proche de T_max quand V_pre dépasse largement V_T (pendant
          un potentiel d'action).
        """
        return self.T_max / (1.0 + np.exp(-(V_pre - self.V_T) / self.K_p))

    def current(self, V_post, s):
        """
        Courant synaptique reçu par le neurone postsynaptique, à
        l'instant où sa tension vaut V_post et où la synapse est dans
        l'état s.

        Comme pour les canaux ioniques classiques, ce courant s'annule
        quand V_post = E_syn (pas de force motrice) et change de signe
        de part et d'autre.
        """
        return self.g_syn * s * (V_post - self.E_syn)

    def ds_dt(self, V_pre, s):
        """
        Dérivée temporelle de la variable synaptique s, étant donné
        la tension présynaptique V_pre et l'état actuel s.

        Comme pour les portes m/h/n, on retrouve la structure
        "vitesse d'ouverture * fraction fermée - vitesse de fermeture
        * fraction ouverte", mais ici la 'vitesse d'ouverture' dépend
        de la libération de neurotransmetteur T(V_pre).
        """
        T_rel = self.transmitter_release(V_pre)
        return self.alpha * T_rel * (1 - s) - self.beta * s


class Network:
    """
    Assemble une liste de neurones (HHNeuron) et de synapses (Synapse)
    en un seul système d'équations différentielles, exploitable par
    scipy.integrate.solve_ivp.

    Organisation du vecteur d'état plat `y` :
        [V1, m1, h1, n1,   <- neurone 0
         V2, m2, h2, n2,   <- neurone 1
         ...,
         s_0, s_1, ...]    <- une valeur de s par synapse, à la fin

    Cette classe ne fait "que" de la comptabilité : elle ne redéfinit
    aucune équation biophysique, elle se contente de router les bonnes
    tensions vers les bonnes synapses, et les bons courants
    synaptiques vers les bons neurones.
    """

    def __init__(self, neurons, synapses):
        """
        neurons  : liste d'objets HHNeuron.
        synapses : liste d'objets Synapse (chacune référence déjà son
                   neurone pré et post via ses attributs .pre/.post).
        """
        self.neurons = neurons        # list of HHNeuron
        self.synapses = synapses      # list of Synapse (pre/post reference neurons)
        # layout of the flat state vector: [V,m,h,n]*len(neurons) + [s]*len(synapses)
        self.n_neurons = len(neurons)
        self.n_synapses = len(synapses)

    def initial_state(self, V0=-65.0):
        """
        Construit le vecteur d'état initial y0, avec tous les
        neurones au repos (V = V0, portes à leurs valeurs d'équilibre)
        et toutes les synapses fermées (s = 0).
        """
        y0 = []
        for neuron in self.neurons:
            m0, h0, n0 = neuron.steady_state(V0)
            y0 += [V0, m0, h0, n0]
        y0 += [0.0] * self.n_synapses   # all synapses start fully closed
        return np.array(y0)

    def unpack(self, y):
        """
        Découpe le vecteur d'état plat y en :
        - une liste de tuples (V, m, h, n), un par neurone,
        - un tableau des valeurs de s, une par synapse.

        C'est l'opération inverse de la construction faite dans
        initial_state / rhs.
        """
        neuron_states = []
        for i in range(self.n_neurons):
            neuron_states.append(y[4 * i: 4 * i + 4])   # (V, m, h, n)
        syn_states = y[4 * self.n_neurons:]
        return neuron_states, syn_states

    def rhs(self, t, y):
        """
        Fonction "right-hand side" attendue par solve_ivp : étant
        donné le temps t et l'état complet y, renvoie dy/dt.

        Étapes :
        1. On sépare y en états individuels par neurone et par synapse.
        2. Pour chaque synapse, on calcule le courant qu'elle injecte
           dans son neurone postsynaptique (en lisant la tension du
           neurone présynaptique), et sa propre dérivée ds/dt.
        3. Pour chaque neurone, on calcule ses dérivées (dV, dm, dh, dn)
           en tenant compte de la somme des courants synaptiques reçus.
        4. On recolle tout dans un seul vecteur dy/dt, dans le même
           ordre que le vecteur d'état y.
        """
        neuron_states, syn_states = self.unpack(y)
        V = [state[0] for state in neuron_states]  # voltages, indexed like self.neurons

        # synaptic currents injected into each neuron (sum if several synapses converge)
        I_syn_per_neuron = [0.0] * self.n_neurons
        ds_list = []
        for syn, s in zip(self.synapses, syn_states):
            i_pre = self.neurons.index(syn.pre)
            i_post = self.neurons.index(syn.post)
            I_syn_per_neuron[i_post] += syn.current(V[i_post], s)
            ds_list.append(syn.ds_dt(V[i_pre], s))

        dydt = []
        for i, (neuron, state) in enumerate(zip(self.neurons, neuron_states)):
            Vi, mi, hi, ni = state
            dV, dm, dh, dn = neuron.derivatives(t, Vi, mi, hi, ni, I_syn=I_syn_per_neuron[i])
            dydt += [dV, dm, dh, dn]
        dydt += ds_list
        return dydt

    def simulate(self, t_span, V0=-65.0, **solve_ivp_kwargs):
        """
        Intègre le système sur l'intervalle de temps t_span = (t0, tf).

        V0 : tension de repos utilisée pour initialiser tous les
             neurones (voir initial_state).
        **solve_ivp_kwargs : arguments supplémentaires transmis
             directement à scipy.integrate.solve_ivp (ex : t_eval,
             method, rtol, atol, max_step...).
        """
        y0 = self.initial_state(V0)
        sol = solve_ivp(self.rhs, t_span, y0, **solve_ivp_kwargs)
        return sol

    def voltages(self, sol):
        """
        Extrait, à partir du résultat sol de solve_ivp, un dictionnaire
        {nom_du_neurone: V(t)} pratique pour le tracé des résultats.
        """
        return {neuron.name: sol.y[4 * i] for i, neuron in enumerate(self.neurons)}


# ----------------------------------------------------------------------
# Exemple d'utilisation :
# - neurone1 reçoit un courant en créneau et décharge des potentiels
#   d'action ;
# - neurone2 ne reçoit aucun courant externe : il n'est excité que via
#   une synapse excitatrice provenant de neurone1.
# ----------------------------------------------------------------------

def step_current(t):
    """Courant en créneau : 10 uA/cm^2 entre 5 et 80 ms, 0 sinon."""
    return 10.0 if 5.0 <= t <= 80.0 else 0.0

neuron1 = HHNeuron(I_ext=step_current, name="neuron1")
neuron2 = HHNeuron(I_ext=None, name="neuron2")   # no direct drive

synapse_1_to_2 = Synapse(g_syn=0.5, E_syn=0.0, pre=neuron1, post=neuron2)

net = Network(neurons=[neuron1, neuron2], synapses=[synapse_1_to_2])

sol = net.simulate(t_span = (0.0, 100.0),
                    t_eval = np.linspace(0, 100, 20000),
                    method = "RK45", max_step=0.02, rtol=1e-8, atol=1e-8)

V = net.voltages(sol)
s12 = sol.y[8]   # l'état de la synapse est la dernière ligne du vecteur d'état
t = sol.t

# --- tracé des résultats ---
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

axes[0].plot(t, V["neuron1"], color="tab:blue")
axes[0].set_ylabel("V1 (mV)")
axes[0].set_title("Neurone présynaptique (stimulé par un courant en créneau)")

axes[1].plot(t, V["neuron2"], color="tab:red")
axes[1].set_ylabel("V2 (mV)")
axes[1].set_title("Neurone postsynaptique (excité uniquement via la synapse)")

axes[2].plot(t, s12, color="tab:green")
axes[2].set_ylabel("s (variable synaptique)")
axes[2].set_xlabel("temps (ms)")
axes[2].set_title("Variable de porte synaptique s12(t)")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
plt.show()