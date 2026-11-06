import math
import numpy as np
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

def create_dicke_state(N: int, K: int) -> QuantumCircuit:
    qc = QuantumCircuit(N, name=f"Dicke_{N}_{K}")
    dim = 1 << N
    statevector = np.zeros(dim, dtype=complex)
    
    valid_indices = []
    for idx in range(dim):
        if bin(idx).count("1") == K:
            valid_indices.append(idx)
            
    norm = 1.0 / math.sqrt(len(valid_indices))
    for idx in valid_indices:
        statevector[idx] = norm
        
    sp = StatePreparation(statevector)
    qc.append(sp, list(range(N)))
    return qc

def build_xy_mixer(
    N: int,
    beta: float,
    connectivity: str = "ring"
) -> QuantumCircuit:
    qc = QuantumCircuit(N, name="XY_Mixer")
    
    even_pairs = [(i, i + 1) for i in range(0, N - 1, 2)]
    odd_pairs = [(i, i + 1) for i in range(1, N - 1, 2)]
    
    if connectivity == "ring" and N > 2:
        if (N - 1) % 2 == 1:
            odd_pairs.append((N - 1, 0))
        else:
            even_pairs.append((N - 1, 0))

    for i, j in even_pairs:
        qc.rxx(beta, i, j)
        qc.ryy(beta, i, j)
        
    for i, j in odd_pairs:
        qc.rxx(beta, i, j)
        qc.ryy(beta, i, j)

    return qc
