import time
from bees_algorithm import BeesAlgorithm
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from data import get_data
from pyswarms.single.global_best import GlobalBestPSO


def evaluate_meanshift(data, bandwidth):
    result = []
    ms = MeanShift(bandwidth=bandwidth[0], cluster_all=True)
    labels = ms.fit_predict(data)

    result = silhouette_score(data, labels)

    return result


def scatter_meanshift(data, bandwidth):
    ms = MeanShift(bandwidth=bandwidth, cluster_all=True)
    labels = ms.fit_predict(data)
    print(silhouette_score(data, labels))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k')
    plt.title('Mean Shift Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def bee(data):
    def objective_function(bandwidth):
        return evaluate_meanshift(data, bandwidth)

    param_ranges = [[0.1], [10.0]]

    bee_algorithm = BeesAlgorithm(objective_function, param_ranges[0], param_ranges[1], ns=10, nb=5, ne=1, nrb=10, nre=15,
                                  stlim=10)
    _, _ = bee_algorithm.performFullOptimisation(max_iteration=5)
    best = bee_algorithm.best_solution
    print(best.values)
    return best.values


if __name__ == "__main__":
    data = get_data(300)
    start_time = time.time()
    best_bandwidth = bee(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Bee function completed in {elapsed_time:.4f} seconds")
    scatter_meanshift(data, best_bandwidth[0])