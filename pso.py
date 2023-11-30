import time
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from data import get_data
from pyswarms.single.global_best import GlobalBestPSO


def evaluate_meanshift(data, bandwidth):
    result = []
    for i in bandwidth:
        ms = MeanShift(bandwidth=i[0], cluster_all=True)
        labels = ms.fit_predict(data)

        result.append(-1 * silhouette_score(data, labels))

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


def pso(data, swarm_size=10, iterations=5, w=0.75, c1=0.5, c2=0.5):
    def objective_function(bandwidth):
        return evaluate_meanshift(data, bandwidth)

    lower_bound = 0.1
    upper_bound = 10.0
    bounds = [(lower_bound,), (upper_bound,)]

    optimizer = GlobalBestPSO(n_particles=swarm_size, dimensions=1, options={'w': w, 'c1': c1, 'c2': c2}, bounds=bounds)

    best_cost, best_bandwidth = optimizer.optimize(objective_function, iters=iterations)

    return best_bandwidth


if __name__ == "__main__":
    data = get_data(300)
    start_time = time.time()
    best_bandwidth = pso(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Bee function completed in {elapsed_time:.4f} seconds")
    scatter_meanshift(data, best_bandwidth[0])