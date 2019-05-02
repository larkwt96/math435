
"""
kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
centers = kmeans.cluster_centers_
block_centers = self.get_block_centers()
old_shape = block_centers.shape
classes = kmeans.predict(block_centers.reshape(-1, 2))
classes = classes.reshape(*old_shape[:-1])
return classes, n, centers
"""


class ManKMeans:
    def __init__(self, n_clusters, random_state):
        self.K = n_clusters
        self.cluster_centers_ = None

    def assign_cluster(self):
        # assign to closest cluster
        # get most populated
        # get farthest
        #
        # reassign farthest.
        #
        pass

    def regen_cluster(self):
        pass

    def fit(self, X, pops):
        # shape is samp dim
        self.X = X
        self.pops = pops

    def predict(X):
        # shape is samp, dim
        pass
