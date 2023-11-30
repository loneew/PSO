from sklearn.datasets import make_blobs


def get_data(n_samples):
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=42, cluster_std=4)
    return X


if __name__ == "__main__":
    print(get_data(300))
