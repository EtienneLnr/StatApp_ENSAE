# StatApp ENSAE: Few-Shot Learning for Time Series Classification in the Energy Domain

## Project Description

### Supervisors:
- Etienne Le Naour, Tahar Nabil  
- Email: etienne.le-naour@edf.fr, tahar.nabil@edf.fr  

### Problem Statement

Few-shot learning is a machine learning paradigm that enables a model to generalize effectively from a limited number of labeled examples. This approach is particularly relevant for companies like EDF, which collect millions of time series data from sensors, but labeling these datasets can be costly and labor-intensive.

This research project provides an opportunity to explore various classification methods across different observability scenarios (e.g., 1%, 5%, 10%, 20% of observed labels). Students will be encouraged to consider key concepts in time series, such as distance metrics (e.g., Euclidean Distance (ED) or Dynamic Time Warping (DTW)), as well as representation learning. Dimensionality reduction techniques may also be explored for data visualization, including Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).

### Project Scope

The project offers various paths with different levels of difficulty based on student progress:

1. Application of classical machine learning classifiers, such as k-Nearest Neighbors (k-NN) using ED or DTW, and Support Vector Machines (SVM) for various observability scenarios.
2. Construction of simple representations (e.g., descriptive statistics vectors or Symbolic Aggregate approXimation (SAX) representations) followed by the application of classifiers across different observability scenarios.
3. Learning neural representations through contrastive methods, followed by the application of classifiers for diverse observability scenarios.

## Project Structure

```
.
├── README.md
├── datasets
│   ├── Elec
│   │   └── ECL.csv
│   └── FordA
│       ├── X_test.npy
│       ├── X_train.npy
│       ├── y_test.npy
│       └── y_train.npy
├── outputs
│   └── figures
│       └── Elec_1-nn_accuracies_per_percentages.png
└── src
    ├── experiments
    │   └── first_expe.py
    └── utils
        ├── make_elec_dataset.py
        └── plot_functions.py
```

## Running the Project

To run the first experiment, navigate to the project root directory and execute the following command:

```bash
python src/experiments/first_expe.py
```


### Available Datasets

Two datasets are provided for the students:

- **Electricity**: A dataset of time series data representing household electricity consumption in Portugal. Class 0 indicates winter, and class 1 indicates summer. You can access the dataset [here](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR).

- **FordA**: A dataset containing 500 time steps of measurements of engine noise. The classification problem involves diagnosing whether a certain symptom exists or does not exist in an automotive subsystem. You can access the dataset [here](https://www.timeseriesclassification.com/description.php?Dataset=FordA).

Other datasets may also be considered for additional experiments.

