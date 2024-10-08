import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
def generate_ga_fitness(num_generations, num_improvements, noise_level=0.01):
    # Generate base fitness curve
    fitness = np.zeros(num_generations)
    saturation = np.random.uniform(0.3,0.8)
    improvement_points = np.sort(np.random.choice(int(num_generations*saturation), num_improvements, replace=False))
    # improvement_points = np.floor(np.log10(improvement_points + 1)*100).astype(int)

    current_fitness = 0
    for point in improvement_points:
        improvement = np.random.uniform(0.01, 0.2)  # Random improvement between 1% and 20%
        fitness[point:] = current_fitness + improvement
        current_fitness += improvement
        idx = np.random.randint(0,10)
        improvement_2 = np.random.uniform(0.01, 0.04)  # Random improvement between 1% and 20%
        fitness[point+idx:] = current_fitness + improvement_2
        current_fitness += improvement_2
        
    
    # Add noise to simulate population variance
    # noise = np.random.normal(0, noise_level, num_generations)
    # fitness += noise
    
    # Ensure fitness is between 0 and 1
    fitness = fitness/np.max(fitness)
    
    return fitness

# Set up the plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
LW = 2
num_generations = 15000
plt.figure(figsize=(12, 6))
iter = np.arange(num_generations)
log_iter = iter
n_improvements_1 = 15
# Generate and plot data for best fitness
fitness_1 = generate_ga_fitness(num_generations, n_improvements_1, noise_level=0.005)
plt.plot(log_iter, 0.63*fitness_1, label='Run 1', color='r', linestyle='-.', linewidth=LW)

# Generate and plot data for average fitness
n_improvements_2 = 10
fitness_2 = generate_ga_fitness(num_generations, n_improvements_2, noise_level=0.02)
fitness_2 = 0.58*fitness_2 
plt.plot(log_iter, fitness_2, label='Run 2', color='b', linestyle='--', linewidth=LW)

n_improvements_3 = 14
fitness_3 = generate_ga_fitness(num_generations, n_improvements_3, noise_level=0.01)
fitness_3 = fitness_3 
plt.plot(log_iter, fitness_3, label='Run 3', color='g', linewidth=LW)
# Add a fill between average and best fitness
# plt.fill_between(range(num_generations), avg_fitness, best_fitness, color='gray', alpha=0.3)

# plt.title('Genetic Algorithm Learning Curve')
plt.xlabel('Iterations')
# plt.xscale('log')
plt.ylabel('Normalized score')
plt.legend()
# plt.grid(True)
plt.show()