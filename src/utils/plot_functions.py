import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[3]))

import matplotlib.pyplot as plt 


def plot_accuracies_per_percentages(percentages, accuracies, name):

    plt.figure(figsize=(10, 5))
    plt.plot(percentages, accuracies, marker='o', linestyle='-', color='b',
             label="Accuracy according to percentage of X_train observed")
    plt.xlabel("Percentage of X_train observed")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
               fancybox=True, shadow=True, ncol=1)
    plt.savefig(f"outputs/figures/{name}_accuracies_per_percentages.png", bbox_inches='tight')