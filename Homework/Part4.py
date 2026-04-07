import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

def ackley(x):
    x1, x2 = x[0], x[1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return term1 + term2 + np.e + 20

# 2. Adapted Custom Nelder-Mead
def custom_nelder_mead(func, x0, l=1.5, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, max_iter=100, tol=1e-6):
    n = len(x0)
    t1 = (l / (n * np.sqrt(2))) * (np.sqrt(n + 1) - 1)
    t2 = l / np.sqrt(2)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        s_i = np.full(n, t1); s_i[i] += t2
        simplex.append(x0 + s_i)
    
    simplex = np.array(simplex)
    history = []
    func_calls = 0

    for it in range(max_iter):
        res = np.array([func(x) for x in simplex])
        func_calls += len(simplex)
        idx = np.argsort(res)
        simplex, res = simplex[idx], res[idx]
        
        x_c = np.mean(simplex[:-1], axis=0)
        x_r = x_c + alpha * (x_c - simplex[-1])
        f_r = func(x_r)
        func_calls += 1
        
        move, target_pt = "", None
        
        if res[0] <= f_r < res[-2]:
            simplex[-1] = x_r
        elif f_r < res[0]:
            x_e = x_c + gamma * (x_c - simplex[-1])
            f_e = func(x_e)
            func_calls += 1
            if f_e < f_r:
                simplex[-1] = x_e
            else:
                simplex[-1] = x_r
        else:
            if f_r < res[-1]:
                x_oc = x_c + rho * (x_c - simplex[-1])
                func_calls += 1
                if func(x_oc) <= f_r:
                    simplex[-1] = x_oc
                else:
                    simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
            else:
                x_ic = x_c - rho * (x_c - simplex[-1])
                func_calls += 1
                if func(x_ic) < res[-1]:
                    simplex[-1] = x_ic
                else:
                    simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])

        history.append({'best_val': res[0], 'best_pt': simplex[0].copy()})
        if np.std(res) < tol: 
            break
            
    return simplex[0], history, func_calls


# --- 1. Particle Swarm Optimization (PSO) ---
def particle_swarm_optimize(func, pop_size=30, iterations=50, w=0.5, c1=1.5, c2=1.5):
    particles = np.random.uniform(-5, 5, (pop_size, 2))
    velocities = np.zeros((pop_size, 2))
    pbest = particles.copy()
    pbest_obj = np.array([func(p) for p in pbest])
    gbest = pbest[np.argmin(pbest_obj)]
    gbest_obj = min(pbest_obj)
    
    history = []
    for _ in range(iterations):
        for i in range(pop_size):
            r1, r2 = np.random.rand(2)
            velocities[i] = w*velocities[i] + c1*r1*(pbest[i]-particles[i]) + c2*r2*(gbest-particles[i])
            particles[i] += velocities[i]
            
            # Keep within bounds
            particles[i] = np.clip(particles[i], -5, 5)
            
            obj = func(particles[i])
            if obj < pbest_obj[i]:
                pbest[i] = particles[i]
                pbest_obj[i] = obj
                if obj < gbest_obj:
                    gbest = particles[i]
                    gbest_obj = obj
        history.append(gbest_obj)
    return gbest, history

# --- 2. Continuous Genetic Algorithm (GA) ---
def genetic_algorithm_optimize(func, pop_size=30, generations=50, mutation_rate=0.1, crossover_rate=0.8):
    pop = np.random.uniform(-5, 5, (pop_size, 2))
    history = []
    
    for _ in range(generations):
        fitness = np.array([func(ind) for ind in pop])
        
        # Track best individual
        best_idx = np.argmin(fitness)
        history.append(fitness[best_idx])
        best_ind = pop[best_idx].copy()
        
        new_pop = []
        for _ in range(pop_size):
            # Tournament Selection
            i, j = np.random.randint(0, pop_size, 2)
            parent1 = pop[i] if fitness[i] < fitness[j] else pop[j]
            i, j = np.random.randint(0, pop_size, 2)
            parent2 = pop[i] if fitness[i] < fitness[j] else pop[j]
            
            # Arithmetic Crossover
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()
                
            # Gaussian Mutation
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.5, 2)
                
            # Keep within bounds
            child = np.clip(child, -5, 5)
            new_pop.append(child)
            
        pop = np.array(new_pop)
        pop[0] = best_ind  # Elitism: carry over the best individual
        
    return best_ind, history

nm_best_pt, nm_history, _ = custom_nelder_mead(ackley, [0.5, 0.5])
pso_best_pt, pso_history = particle_swarm_optimize(ackley)
ga_best_pt, ga_history = genetic_algorithm_optimize(ackley)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
nm_line, = plt.plot([h['best_val'] for h in nm_history], marker='o', label="Nelder-Mead")
pso_line, = plt.plot(pso_history, marker='o', label="Particle Swarm")
ga_line, = plt.plot(ga_history, marker='o', label="Genetic")
plt.legend()
plt.title('Convergence of Custom Gradient Free Optimizers')
plt.xlabel('Iteration')
plt.ylabel('Function Value')

plt.subplot(1, 2, 2)
x_val = np.linspace(-2, 2, 100)
y_val = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_val, y_val)
Z = ackley([X, Y])
plt.contourf(X, Y, Z, levels=30, cmap='viridis')
plt.plot(nm_best_pt[0], nm_best_pt[1], 'o', color=nm_line.get_color(), label='NM Optimal Point')
plt.plot(pso_best_pt[0], pso_best_pt[1], 'o', color=pso_line.get_color(), label='PSO Optimal Point')
plt.plot(ga_best_pt[0], ga_best_pt[1], 'o', color=ga_line.get_color(), label='GA Optimal Point')
plt.title('Contour Plot of Ackley Function')
plt.legend()
plt.tight_layout()
plt.show()

print("Hello World")