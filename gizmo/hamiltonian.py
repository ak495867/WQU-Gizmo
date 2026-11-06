import numpy as np
from typing import Tuple, Dict
from qiskit.quantum_info import SparsePauliOp

def build_cost_hamiltonian(
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float = 1.0
) -> Tuple[SparsePauliOp, np.ndarray, np.ndarray, float]:
    N = len(mu)
    h_vector = np.zeros(N, dtype=float)
    J_matrix = np.zeros((N, N), dtype=float)
    
    for i in range(N):
        h_vector[i] = 0.5 * mu[i] - 0.25 * q * sigma[i, i] - 0.25 * q * (np.sum(sigma[i, :]) - sigma[i, i])
        for j in range(i + 1, N):
            J = 0.25 * q * sigma[i, j]
            J_matrix[i, j] = J
            J_matrix[j, i] = J

    offset = 0.5 * float(np.sum(0.5 * q * np.diag(sigma) - mu))
    for i in range(N):
        for j in range(i + 1, N):
            offset += 0.25 * q * sigma[i, j]

    pauli_list = []
    
    id_label = "I" * N
    if abs(offset) > 1e-12:
        pauli_list.append((id_label, offset))

    for i in range(N):
        if abs(h_vector[i]) > 1e-12:
            chars = ["I"] * N
            chars[N - 1 - i] = "Z"
            pauli_list.append(("".join(chars), h_vector[i]))

    for i in range(N):
        for j in range(i + 1, N):
            if abs(J_matrix[i, j]) > 1e-12:
                chars = ["I"] * N
                chars[N - 1 - i] = "Z"
                chars[N - 1 - j] = "Z"
                pauli_list.append(("".join(chars), J_matrix[i, j]))

    if not pauli_list:
        pauli_list = [(id_label, 0.0)]

    return SparsePauliOp.from_list(pauli_list), h_vector, J_matrix, offset

def build_penalty_hamiltonian(
    N: int,
    K: int,
    penalty_lambda: float = 2.0
) -> SparsePauliOp:
    M = 0.5 * N - float(K)
    offset = penalty_lambda * (M ** 2 + 0.25 * N)
    linear_coeff = -penalty_lambda * M
    quad_coeff = 0.5 * penalty_lambda

    pauli_list = [("I" * N, offset)]

    for i in range(N):
        chars = ["I"] * N
        chars[N - 1 - i] = "Z"
        pauli_list.append(("".join(chars), linear_coeff))

    for i in range(N):
        for j in range(i + 1, N):
            chars = ["I"] * N
            chars[N - 1 - i] = "Z"
            chars[N - 1 - j] = "Z"
            pauli_list.append(("".join(chars), quad_coeff))

    return SparsePauliOp.from_list(pauli_list)
