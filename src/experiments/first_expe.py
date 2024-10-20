import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[3]))

# Standard imports 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Custom imports 
from src.utils.make_elec_dataset import make_dataset
from src.utils.plot_functions import plot_accuracies_per_percentages


# Select Elec or FordA
dataset_name = 'Elec'


if dataset_name == 'Elec':
    elec = pd.read_csv('datasets/Elec/ECL.csv')
    X_train, y_train, X_test, y_test = make_dataset(elec)

elif dataset_name == 'FordA':
    X_train = np.load("datasets/FordA/X_train.npy").squeeze()
    y_train = np.load("datasets/FordA/y_train.npy").squeeze()
    X_test = np.load("datasets/FordA/X_test.npy").squeeze()
    y_test = np.load("datasets/FordA/y_test.npy").squeeze()

else:
    raise NotImplementedError

# Define a list of observed percentages

percentages_l = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
accuracies_l = []


for pct in percentages_l:

    num_samples = int(len(X_train) * pct / 100)
    
    # Subsampling 
    X_train_subset = X_train[:num_samples]
    y_train_subset = y_train[:num_samples]
    
    # Instantiate model
    model = KNeighborsClassifier(n_neighbors=1)

    # Training
    model.fit(X_train_subset, y_train_subset)
    
    # Inference
    y_pred = model.predict(X_test)
    
    # Compute test accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_l.append(accuracy)

    print(f"Percentage: {pct}/100 \n  \
           Nb samples observed: {num_samples} \n \
          Accuracy: {accuracy:.3f}")

# Plot the accuracy as a function of the percentage of data observed
plot_accuracies_per_percentages(
                    percentages=percentages_l,
                    accuracies=accuracies_l, 
                    name=f"{dataset_name}_1-nn"
                )