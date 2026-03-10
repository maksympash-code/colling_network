import numpy as np
from src.presentation.viz_temp import plot_temperature

# приклад: якась T матриця
T = np.random.uniform(25, 80, size=(10, 10))
plot_temperature(T, title="Demo temperature 10x10", annotate=True)