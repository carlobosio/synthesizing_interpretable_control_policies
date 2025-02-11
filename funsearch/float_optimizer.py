from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from zoopt import Dimension, Objective, Parameter, Opt, Solution
from float_extractor import ProgramWrapper
from sandbox import DummySandbox

class FloatOptimizer:
    def __init__(
        self, 
        program: str,
        evaluate_fn: Callable[[str], float],  # Function that evaluates a program and returns score
        bounds: List[Tuple[float, float]] = None,
        budget: int = 100,
        maximize_score: bool = True,
    ):
        """Initialize optimizer for floating point parameters in a program.
        
        Args:
            program: String containing the code snippet to optimize
            evaluate_fn: Function that takes a program string and returns its score
            bounds: List of (min, max) bounds for each parameter
            budget: Number of evaluations allowed for optimization
            maximize_score: If True, maximize score; if False, minimize score
        """
        self.program_wrapper = ProgramWrapper(program)
        self.evaluate_fn = evaluate_fn
        self.maximize_score = maximize_score
        self.num_params = self.program_wrapper.get_num_floats()
        
        # Set default bounds if none provided
        if bounds is None:
            self.bounds = [(-10.0, 10.0)] * self.num_params
        else:
            if len(bounds) != self.num_params:
                raise ValueError(f"Expected {self.num_params} bounds, got {len(bounds)}")
            self.bounds = bounds
            
        self.budget = budget
        self.optimization_history = []
        self.best_so_far = float('-inf') if maximize_score else float('inf')

    def _objective_function(self, solution: List[float]) -> float:
        """Compute score using provided evaluation function."""
        if hasattr(solution, 'get_x'):
            solution = solution.get_x()
            
        # Update program with new parameters
        self.program_wrapper.sub_floats(solution)
        program = self.program_wrapper.get_program()
        
        # Get score using evaluation function
        score = self.evaluate_fn(program)
        
        # Track best score
        if self.maximize_score:
            self.best_so_far = max(self.best_so_far, score)
        else:
            self.best_so_far = min(self.best_so_far, score)
        self.optimization_history.append(self.best_so_far)
        
        return -score if self.maximize_score else score

    def plot_optimization_history(self, save_path: str = None):
        """Plot the optimization history showing best score over iterations.
        
        Args:
            save_path: Optional path to save the plot. If None, displays the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.optimization_history, 'b-', label='Best score')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (log scale)')
        plt.title('Optimization Progress')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def optimize(self) -> Tuple[List[float], float]:
        """Run optimization to find best parameters.
        
        Returns:
            Tuple of (best_parameters, best_score)
        """
        # Reset history for new optimization run
        self.optimization_history = []
        self.best_so_far = float('-inf') if self.maximize_score else float('inf')
        
        dim_size = self.num_params
        dim_regs = [[-10.0, 10.0]] * dim_size  # search range for each parameter
        dim_tys = [True] * dim_size  # continuous for all parameters
        
        dimension = Dimension(dim_size, dim_regs, dim_tys)
        objective = Objective(self._objective_function, dimension)
        
        # Configure optimization parameters
        parameter = Parameter(
            budget=self.budget,
            sequential=True,
            parallel=False,
            noise_handling=True
        )
        
        solution = Opt.min(objective, parameter)
        return solution.get_x(), solution.get_value()

def example_usage():
    # Example program with parameters to optimize
    program = """
def function(inputs) -> float:
    x1 = inputs['x1']
    x2 = inputs['x2']
    return 3.5*x1**2 + 5.5*x2**2
"""
    
    # Generate synthetic training data
    input_score_pairs = []
    for _ in range(10):
        x1 = np.random.uniform(-2, 2)
        x2 = np.random.uniform(-2, 2)
        true_output = 2.0 * x1**2 + 3.0 * x2**2
        input_score_pairs.append(({'x1': x1, 'x2': x2}, true_output))
    
    # Create optimizer and run optimization
    sandbox = DummySandbox()
    
    # Define evaluation function that computes MSE across all training points
    def evaluate_fn(program):
        mse = 0.0
        for inputs, true_output in input_score_pairs:
            predicted = sandbox.run(program, "function", inputs, 30)
            mse += (predicted - true_output) ** 2
        return -mse / len(input_score_pairs)  # Negative because we're maximizing
    
    optimizer = FloatOptimizer(
        program=program,
        evaluate_fn=evaluate_fn,
        bounds=[(-5, 5), (-5, 5)],
        budget=300,
        maximize_score=True
    )
    
    best_params, best_score = optimizer.optimize()
    print(f"Optimized parameters: {best_params}")
    print(f"Final MSE: {best_score}")
    
    # Plot and save the optimization history
    optimizer.plot_optimization_history()

if __name__ == "__main__":
    example_usage() 