import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from .subspace import create_dicke_state, build_xy_mixer
from .swap_network import build_linear_swap_network_cost_layer
from .classical import evaluate_portfolio_cost

def build_naive_qaoa_circuit(
    total_hamiltonian: SparsePauliOp,
    p: int,
    gammas: List[float],
    betas: List[float]
) -> QuantumCircuit:
    N = total_hamiltonian.num_qubits
    qc = QuantumCircuit(N, N)
    
    for i in range(N):
        qc.h(i)
        
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        for pauli_str, coeff in total_hamiltonian.to_list():
            coeff_val = float(np.real(coeff))
            if abs(coeff_val) < 1e-12:
                continue
            z_indices = [N - 1 - idx for idx, char in enumerate(pauli_str) if char == "Z"]
            if len(z_indices) == 1:
                qc.rz(2.0 * gamma * coeff_val, z_indices[0])
            elif len(z_indices) == 2:
                qc.rzz(2.0 * gamma * coeff_val, z_indices[0], z_indices[1])
                
        for i in range(N):
            qc.rx(2.0 * beta, i)
            
    qc.measure(list(range(N)), list(range(N)))
    return qc

def build_subspace_qaoa_circuit(
    N: int,
    K: int,
    p: int,
    gammas: List[float],
    betas: List[float],
    J_matrix: np.ndarray,
    h_vector: np.ndarray,
    use_swap_network: bool = True
) -> QuantumCircuit:
    qc = QuantumCircuit(N, N)
    
    dicke_circ = create_dicke_state(N, K)
    qc.compose(dicke_circ, inplace=True)
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        if use_swap_network:
            cost_layer = build_linear_swap_network_cost_layer(N, gamma, J_matrix, h_vector)
            qc.compose(cost_layer, inplace=True)
        else:
            for i in range(N):
                if abs(h_vector[i]) > 1e-12:
                    qc.rz(2.0 * gamma * h_vector[i], i)
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(J_matrix[i, j]) > 1e-12:
                        qc.rzz(2.0 * gamma * J_matrix[i, j], i, j)
                        
        mixer_layer = build_xy_mixer(N, beta, connectivity="ring")
        qc.compose(mixer_layer, inplace=True)
        
    qc.measure(list(range(N)), list(range(N)))
    return qc

def evaluate_counts_feasibility(
    counts: Dict[str, int],
    K: int,
    N: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float
) -> Dict[str, Any]:
    total_shots = sum(counts.values())
    valid_shots = 0
    valid_counts = {}
    
    best_feasible_bitstring = None
    best_feasible_cost = float("inf")
    
    weighted_cost_sum = 0.0
    
    for raw_bitstring, count in counts.items():
        bitstring = raw_bitstring.replace(" ", "")
        ones = bitstring.count("1")
        if ones == K:
            valid_shots += count
            valid_counts[bitstring] = count
            
            x = np.array([int(bitstring[N - 1 - i]) for i in range(N)], dtype=float)
            cost = evaluate_portfolio_cost(x, mu, sigma, q)
            weighted_cost_sum += cost * count
            
            if cost < best_feasible_cost:
                best_feasible_cost = cost
                best_feasible_bitstring = bitstring

    feasibility_ratio = valid_shots / total_shots if total_shots > 0 else 0.0
    avg_feasible_cost = weighted_cost_sum / valid_shots if valid_shots > 0 else float("nan")
    
    return {
        "total_shots": total_shots,
        "valid_shots": valid_shots,
        "feasibility_ratio": feasibility_ratio,
        "best_feasible_bitstring": best_feasible_bitstring,
        "best_feasible_cost": best_feasible_cost,
        "average_feasible_cost": avg_feasible_cost,
        "valid_counts": valid_counts
    }

def run_circuit_sampling(
    circuit: QuantumCircuit,
    shots: int = 2048,
    seed: int = 42
) -> Dict[str, int]:
    try:
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler(default_shots=shots, seed=seed)
        job = sampler.run([circuit])
        result = job.result()
        data_bin = result[0].data
        if hasattr(data_bin, "c"):
            return data_bin.c.get_counts()
        first_key = list(data_bin.keys())[0]
        return getattr(data_bin, first_key).get_counts()
    except Exception:
        from qiskit.quantum_info import Statevector
        qc_no_meas = circuit.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = sv.probabilities_dict()
        samples = np.random.default_rng(seed).choice(
            list(probs.keys()),
            size=shots,
            p=list(probs.values())
        )
        counts = {}
        for s in samples:
            counts[s] = counts.get(s, 0) + 1
        return counts

def run_qaoa_optimization(
    cost_func_builder,
    p: int = 1,
    maxiter: int = 35,
    seed: int = 42
) -> Tuple[List[float], List[float], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(0.1, 0.8, size=2 * p)
    
    def loss_objective(params):
        gammas = params[:p]
        betas = params[p:]
        qc = cost_func_builder(gammas, betas)
        counts = run_circuit_sampling(qc, shots=1024, seed=seed)
        shots = sum(counts.values())
        return sum(cost * (cnt / shots) for cost, cnt in [
            (cost_func_builder.eval_bitstring(b), c) for b, c in counts.items()
        ])
        
    res = minimize(
        loss_objective,
        initial_point,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False}
    )
    
    opt_gammas = list(res.x[:p])
    opt_betas = list(res.x[p:])
    final_qc = cost_func_builder(opt_gammas, opt_betas)
    final_counts = run_circuit_sampling(final_qc, shots=4096, seed=seed)
    
    return opt_gammas, opt_betas, final_counts
