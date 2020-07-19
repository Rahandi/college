import math

class Hierarchical:
    def __init__(self, num_cluster):
        self.distance = []
        self.ncluster = num_cluster
        self.cluster = []
        self.cluster_point = [[] for a in range(num_cluster)]

    def fit(self, X):
        self.distance = [[] for a in range(len(X))]
        for a in range(len(X)):
            self.distance[a] = [self._calculate_distance(X[a], X[b]) for b in range(len(X))]
            self.cluster.append([a])
        while len(self.cluster) != self.ncluster:
            cluster_distances = [self._calculate_cluster_distance(self.cluster[a]) for a in range(len(self.cluster))]
            minimum = 100000000
            first_index = 0
            second_index = 0
            for a in range(len(cluster_distances)):
                if self.cluster[a] != []:
                    minimum_nonzero_value = min(cluster_distances[a])
                    minimum_nonzero_value_index = cluster_distances[a].index(minimum_nonzero_value)
                    if minimum_nonzero_value < minimum:
                        minimum = minimum_nonzero_value
                        first_index = minimum_nonzero_value_index
                        second_index = a
            self._merge_point(first_index, second_index)
            cluster_distances = self._update_distance(cluster_distances, minimum_nonzero_value_index)
            self.cluster = [self.cluster[a] for a in range(len(self.cluster)) if self.cluster[a] != []]
        for a in range(len(self.cluster)):
            self.cluster_point[a] = [X[b] for b in self.cluster[a]]
        return self.cluster

    def sse(self):
        jumlah = 0
        for a in range(len(self.cluster_point)):
            centroids = self._get_centroids(self.cluster_point[a])
            for b in range(len(self.cluster_point[a])):
                jumlah += sum([math.pow(centroids[c]-self.cluster_point[a][b][c],2) for c in range(len(centroids))])
        return jumlah

    def _get_centroids(self, cluster):
        centroids = []
        for a in range(len(cluster[0])):
            hasil = sum([b[a] for b in cluster])/len(cluster)
            centroids.append(hasil)
        return centroids

    def _calculate_distance(self, first, second):
        try:
            result = math.pow(sum([math.pow(first[a]-second[a], 2) for a in range(len(first))]), 1/2)
            return result
        except Exception as e:
            raise(e)

    def _calculate_cluster_distance(self, cluster):
        cluster_distance = [self._minimum_distance_between_cluster(cluster, self.cluster[b]) if self._minimum_distance_between_cluster(cluster, self.cluster[b]) != 0 else 1000000 for b in range(len(self.cluster))]
        return cluster_distance

    def _merge_point(self, first_cluster, second_cluster):
        self.cluster[first_cluster].extend(self.cluster[second_cluster])
        self.cluster[second_cluster] = []

    def _minimum_distance_between_cluster(self, first_cluster, second_cluster):
        distance = []
        for a in range(len(first_cluster)):
            for b in range(len(second_cluster)):
                distance.append(self.distance[first_cluster[a]][second_cluster[b]])
        return min(distance)

    def _update_distance(self, cluster_distance, update_index):
        for a in range(len(cluster_distance)):
            cluster_distance[a][update_index] = 1000000
        return cluster_distance