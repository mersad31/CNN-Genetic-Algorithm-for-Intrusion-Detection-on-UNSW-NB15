# FILE: genetic_algorithm.py

import numpy as np
import random
from sklearn.model_selection import train_test_split

# --- 1. GENETIC ALGORITHM CONFIGURATION ---
POPULATION_SIZE = 10     # Number of models in each generation
N_GENERATIONS = 10      # Number of generations to evolve
MUTATION_RATE = 0.1    # Probability of a random change
CROSSOVER_RATE = 0.8   # Probability of breeding
N_EPOCHS_FITNESS = 3   # Train for only a few epochs to quickly evaluate fitness

# --- 2. HYPERPARAMETER SEARCH SPACE (THE 'GENES') ---
HYPERPARAMETER_SPACE = {
    'learning_rate': [0.01, 0.005, 0.001],
    'filters1': [16, 32, 64],
    'filters2': [32, 64, 128],
    'kernel_size': [3, 5],
    'activation_str': ['relu', 'leaky_relu'],
    'dropout_rate': [0.3, 0.4, 0.5],
    'optimizer_str': ['adam', 'rmsprop']
}

# --- 3. LOAD AND PREPARE DATA ONCE ---
print("Loading and preprocessing data for GA...")
X, y, _, _ = preprocess_unsw_nb15()

# Create a smaller validation set from the training data for faster fitness evaluation
# Using 20% of the data for validation during the GA process
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
INPUT_SHAPE = X_train.shape[1:]

# --- 4. GENETIC ALGORITHM CORE FUNCTIONS ---

def create_individual():
    """Creates a random set of hyperparameters (a 'chromosome')."""
    return {key: random.choice(values) for key, values in HYPERPARAMETER_SPACE.items()}

def calculate_fitness(individual):
    """Trains a model with the given hyperparameters and returns its accuracy."""
    print(f"Testing individual: {individual}")
    try:
        model = create_cnn_model(input_shape=INPUT_SHAPE, **individual)

        # Train on the training subset for a few epochs
        history = model.fit(X_train, y_train,
                            epochs=N_EPOCHS_FITNESS,
                            batch_size=256,
                            validation_data=(X_val, y_val),
                            verbose=0) # Set to 0 to keep the log clean

        # Fitness is the validation accuracy on the last epoch
        fitness = history.history['val_accuracy'][-1]
        print(f"-> Fitness (Accuracy): {fitness:.4f}")

    except Exception as e:
        print(f"Error during fitness calculation for {individual}: {e}")
        fitness = 0 # Penalize individuals that cause errors

    return fitness

def crossover(parent1, parent2):
    """Creates a child by combining genes from two parents."""
    child = {}
    for key in HYPERPARAMETER_SPACE.keys():
        # Randomly choose which parent's gene to inherit
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual):
    """Randomly changes one gene in an individual."""
    for key in individual:
        if random.random() < MUTATION_RATE:
            individual[key] = random.choice(HYPERPARAMETER_SPACE[key])
    return individual

# --- 5. MAIN GENETIC ALGORITHM LOOP ---

if __name__ == '__main__':
    # Initialize the first generation
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    for generation in range(N_GENERATIONS):
        print(f"\n{'='*20} GENERATION {generation + 1}/{N_GENERATIONS} {'='*20}")

        # Calculate fitness for the entire population
        fitness_scores = [calculate_fitness(ind) for ind in population]

        # Select the best individuals for the next generation (Elitism)
        best_indices = np.argsort(fitness_scores)[-POPULATION_SIZE//2:] # Keep the top 50%
        next_population = [population[i] for i in best_indices]

        # Create new offspring through crossover and mutation
        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(next_population, k=2) # Select parents from the best
            if random.random() < CROSSOVER_RATE:
                child = crossover(parent1, parent2)
                child = mutate(child)
                next_population.append(child)
            else:
                # If no crossover, just mutate one of the best and add it
                next_population.append(mutate(random.choice(next_population)))

        population = next_population

        # Report the best individual of the current generation
        best_fitness_so_far = max(fitness_scores)
        best_individual_so_far = population[np.argmax(fitness_scores)]
        print(f"\nBest fitness in generation {generation + 1}: {best_fitness_so_far:.4f}")
        print(f"Best hyperparameters: {best_individual_so_far}")

    # Final result
    print("\n--- Genetic Algorithm Finished ---")
    final_fitness_scores = [calculate_fitness(ind) for ind in population]
    best_overall_individual = population[np.argmax(final_fitness_scores)]
    print("\nBest Hyperparameters Found:")
    print(best_overall_individual)