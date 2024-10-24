import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.providers import BackendV2 as Backend
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def calc_solutions_exactes(hamiltonien: SparsePauliOp) -> tuple[float, list[str]]:
    """Calcul classique des solutions de l'hamiltonien fourni en entrée.
    Cela est fait en diagonalisant la matrice de l'hamiltonien.

    Args:
        hamiltonien (SparsePauliOp): Hamiltonien à diagonaliser exprimé sous la forme d'une somme de chaines de Pauli.

    Returns:
        tuple[float, list[str]]:
            - Coût minimal obtenue (float)
            - Liste de solutions binaires associées au coût minimal
    """
    # Transformer l'hamiltonien en matrice
    mat_hamiltonien = np.array(hamiltonien.to_matrix())
    # Diagonaliser la matrice pour en extraire vecteurs et valeurs propres
    valeurs_propres, vecteurs_propres = np.linalg.eig(mat_hamiltonien)

    # Index des valeurs propres minimales
    min_val_propre = np.where(valeurs_propres == np.min(valeurs_propres))[0]
    # Solutions minimales associées aux valeurs propres minimales
    sol_binaires = [bin(idx).lstrip("-0b").zfill(hamiltonien.num_qubits) for idx in min_val_propre]

    # Coût and chaines binaires de toutes les meilleures solutions
    return valeurs_propres[min_val_propre][0].real, sol_binaires


def calc_score(
    params: np.ndarray, num_couches: int, hamiltonien: SparsePauliOp, backend: Backend
) -> tuple[float, float]:
    """Calcul du score attribué aux paramètres optimaux donnés en entrée, selon un circuit quantique avec un nombre de
    couches donné.

    Args:
        params (np.ndarray): Paramètres optimaux trouvés.
        num_couches (int): Nombre de couches pour la construction du circuit de QAOA.
        hamiltonien (SparsePauliOp): Hamiltonien du problème (fonction de coût).
        backend (Backend): Backend utilisé pour créer un Estimateur et un Sampler.

    Returns:
        tuple[float, float]: Cout optimal et score (pourcentage) de la solution trouvée.
    """
    # Construction du circuit de QAOA
    circuit = QAOAAnsatz(hamiltonien, reps=num_couches)
    # Calcul des solutions exactes à des fins de comparaison
    _, sol_binaires = calc_solutions_exactes(hamiltonien)

    # Production de la distribution de probabilité avec les paramètres optimaux fournis
    sampler = Sampler(mode=backend)
    circuit_copie = circuit.decompose(reps=2).copy()
    circuit_copie.measure_all()
    data = sampler.run([(circuit_copie, params)]).result()[0].data.meas
    comptes = data.get_counts()
    nb_shots = data.num_shots

    # Calcul du score (pourcentage de bonnes solutions obtenues sur le nombre de shots d'execution du circuit)
    score = 0
    for sol in sol_binaires:
        if sol in comptes:
            score += comptes[sol]
    score /= nb_shots
    score *= 100.0

    # Calcul de la valeur moyenne (cout) obtenu avec les paramètres optimaux
    estimator = Estimator(mode=backend)
    pm = generate_preset_pass_manager(backend=estimator._backend, optimization_level=1)
    isa_psi = pm.run(circuit)
    isa_observables = hamiltonien.apply_layout(isa_psi.layout)
    cout = estimator.run([(isa_psi, isa_observables, params)]).result()[0].data.evs

    print("Cout optimal : ", cout)
    print("Score : ", score)
    return cout, score


def sauvegarder_res(nom_fichier: str, params: np.ndarray, num_couches: int, hamiltonien: SparsePauliOp):
    """Sauvegarde de vos paramètres et circuit de QAOA optimaux. Le fichier sauvegardé est à soumettre aux juges.

    Args:
        nom_fichier (str): Nom de fichier de sauvegarde. Utilisez un nom significatif pour votre soumission.
        params (np.ndarray): Paramètres optimaux trouvés.
        num_couches (int): Nombre de couches pour conscrtuire le circuit de QAOA.
        hamiltonien (SparsePauliOp): Hamiltonien du problème exprimé sous forme de somme de chaînes de Pauli.
    """
    np.savez(file=nom_fichier, params=params, num_couches=num_couches, hamiltonien=hamiltonien)


def lire_res(nom_fichier: str):
    """Lire les informations sauvegardées avec la fonction plus haut.

    Args:
        nom_fichier (str):  Nom de fichier sauvegardé.

    Returns:
        tuple[float, int, SparsePauliOp]: Paramètres optimaux, nombre de couches de circuit et hamiltonien sauvegardés.
    """
    fichier = np.load(nom_fichier)
    params = fichier["params"]
    num_couches = fichier["num_couches"]
    hamiltonien = fichier["hamiltonien"]

    return params, num_couches, hamiltonien
