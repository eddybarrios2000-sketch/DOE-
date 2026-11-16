import random
import math
import numpy as np

def create_random_tsp(n=9, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))

def calculate_cost(route, coords):
    return sum(np.linalg.norm(coords[route[i]] -
                              coords[route[(i + 1) % len(route)]])
               for i in range(len(route)))

def two_opt_swap(route, i, k):
    new_route = route[:]
    new_route[i:k + 1] = reversed(new_route[i:k + 1])
    return new_route

def simulated_annealing(coords, initial_temp, seed=42, iterations=1500):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords debe contener pares de coordenadas (x, y).")
    if len(coords) < 2:
        raise ValueError("Se necesitan al menos dos ciudades.")
    if not np.isfinite(coords).all():
        raise ValueError("Las coordenadas deben ser valores finitos.")
    if not math.isfinite(initial_temp) or initial_temp <= 0:
        raise ValueError("initial_temp debe ser un numero finito mayor que cero.")
    if not isinstance(iterations, int) or iterations < 2:
        raise ValueError("iterations debe ser un entero mayor o igual que 2.")

    rng = random.Random(seed)
    n = len(coords)
    route = list(range(n))
    rng.shuffle(route)
    current_cost = calculate_cost(route, coords)
    best_route, best_cost = route[:], current_cost
    min_temp = 1e-4
    cooling_rate = (min_temp / initial_temp) ** (1 / (iterations - 1))
    temp = initial_temp

    for _ in range(iterations):
        i, k = sorted(rng.sample(range(n), 2))
        candidate = two_opt_swap(route, i, k)
        candidate_cost = calculate_cost(candidate, coords)
        delta = candidate_cost - current_cost

        if delta < 0 or rng.random() < math.exp(-delta / temp):
            route, current_cost = candidate, candidate_cost
            if current_cost < best_cost:
                best_route, best_cost = route[:], current_cost
        temp *= cooling_rate

    return best_cost, best_route
