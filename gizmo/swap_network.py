import numpy as np
from typing import Tuple, List
from qiskit import QuantumCircuit

def build_linear_swap_network_cost_layer(
    N: int,
    gamma: float,
    J_matrix: np.ndarray,
    h_vector: np.ndarray
) -> Tuple[QuantumCircuit, List[int]]:
    qc = QuantumCircuit(N, name="Swap_Net_Cost")
    
    for i in range(N):
        if abs(h_vector[i]) > 1e-12:
            qc.rz(2.0 * gamma * h_vector[i], i)
            
    loc = list(range(N))
    
    for r in range(N):
        if r % 2 == 0:
            pairs = [(p, p + 1) for p in range(0, N - 1, 2)]
        else:
            pairs = [(p, p + 1) for p in range(1, N - 1, 2)]
            
        for p1, p2 in pairs:
            q_a = loc[p1]
            q_b = loc[p2]
            J_val = J_matrix[q_a, q_b]
            if abs(J_val) > 1e-12:
                qc.rzz(2.0 * gamma * J_val, p1, p2)
                
        if r < N - 1:
            for p1, p2 in pairs:
                qc.swap(p1, p2)
                loc[p1], loc[p2] = loc[p2], loc[p1]
                
    return qc, loc
