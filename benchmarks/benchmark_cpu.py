"""
Banc de test préliminaire mesurant le temps de simulation du solveur CPU
en fonction du nombre de neurones interconnectés.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List

# Assure que la racine du projet est dans sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hodgkin_huxley import HHNeuron
from src.topologies.connectome import Connectome
from src.solvers.cpu_scipy import CPUScipyNetwork


def run_benchmark(sizes: List[int], duration: float = 50.0):
    print(f"=== Début du benchmark CPU (durée simulée: {duration} ms) ===")
    print(f"{'Taille N':<10} | {'Nb Synapses':<12} | {'Temps CPU (s)':<14} | {'Vitesse (ms sim/s)':<18}")
    print("-" * 62)

    for n in sizes:
        # Stimulation sur le premier neurone
        neurons = [
            HHNeuron(
                name=f"N_{i}",
                I_ext=(lambda t: 10.0 if 5.0 <= t <= 40.0 else 0.0) if i == 0 else None,
            )
            for i in range(n)
        ]

        # Topologie en chaîne
        conn = Connectome.make_chain(n_neurons=n, g_syn=0.5)
        net = CPUScipyNetwork.from_connectome(neurons, conn)

        t_start = time.perf_counter()
        _ = net.simulate(
            t_span=(0.0, duration),
            method="RK45",
            max_step=0.05,
            rtol=1e-5,
            atol=1e-5,
        )
        elapsed = time.perf_counter() - t_start

        speed = duration / elapsed if elapsed > 0 else 0.0
        n_edges = len(conn.get_edges())
        print(f"{n:<10} | {n_edges:<12} | {elapsed:<14.3f} | {speed:<18.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CPU pour réseaux Hodgkin-Huxley")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8, 16], help="Tailles de réseaux à évaluer")
    parser.add_argument("--duration", type=float, default=20.0, help="Durée de simulation en ms")
    args = parser.parse_args()

    run_benchmark(sizes=args.sizes, duration=args.duration)
