import time
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from data import get_data
import numpy as np
from numpy.random import default_rng


class FireflyAlgorithm:
    def __init__(self, pop_size=10, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)

    def run(self, function, dim, lb, ub, max_evals):
        fireflies = self.rng.uniform(lb, ub, (self.pop_size, dim))
        intensity = np.apply_along_axis(function, 1, fireflies)
        best_index = np.argmin(intensity)
        best = np.min(intensity)
        best_params = fireflies[best_index]
        evaluations = self.pop_size
        new_alpha = self.alpha
        search_range = ub - lb

        while evaluations <= max_evals:
            new_alpha *= 0.97
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if intensity[i] >= intensity[j]:
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * r)
                        steps = new_alpha * (self.rng.random(dim) - 0.5) * search_range
                        x = fireflies[i]
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        intensity[i] = function(fireflies[i])
                        evaluations += 1
                        if intensity[i] < best:
                            best_params = x
                        best = min(intensity[i], best)
        return (best, best_params)


def evaluate_meanshift(data, bandwidth):
    result = []
    ms = MeanShift(bandwidth=bandwidth[0], cluster_all=True)
    labels = ms.fit_predict(data)

    result = -1 * silhouette_score(data, labels)

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


def firefly(data):
    def objective_function(bandwidth):
        return evaluate_meanshift(data, bandwidth)

    FA = FireflyAlgorithm()
    best, best_params = FA.run(function=objective_function, dim=1, lb=0.1, ub=10.0, max_evals=5)

    print(best_params)

    return best_params


if __name__ == "__main__":
    data = get_data(300)
    start_time = time.time()
    best_bandwidth = firefly(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Bee function completed in {elapsed_time:.4f} seconds")
    scatter_meanshift(data, best_bandwidth[0])