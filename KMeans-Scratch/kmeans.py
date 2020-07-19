import random
import math

class KMeans:
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster

    def fit(self, X, normalize=True, init='random'):
        # initialize centroids
        centroids = self.initialize_centroids(X, init)
        # normalize
        if normalize == True:
            X = self.normalize(X)
        while True:
            centroids_members = [[] for a in range(self.n_cluster)]
            centroids_members_index = [[] for a in range(self.n_cluster)]
            difference = 0
            for a in range(len(X)):
                # calculate distance between data and all centroids
                distances = [self.calculate_distance(X[a], b) for b in centroids]
                # finding closest centroids to data
                closest_centroids = distances.index(min(distances))
                # appending data to corresponding list with closest centroids as index
                centroids_members[closest_centroids].append(X[a])
                centroids_members_index[closest_centroids].append(a)
            # calculate new centroids
            new_centroids = self.calculate_new_centroids(centroids_members)
            difference = sum([self.calculate_distance(new_centroids[a], centroids[a]) for a in range(len(new_centroids))])
            centroids = new_centroids
            if difference == 0:
                break
        self.normalizing = normalize
        self.centroids = centroids
        self.clustered = centroids_members
        Y = [-1] * len(X)
        # create list of clustered data
        for a in range(len(centroids_members_index)):
            for b in centroids_members_index[a]:
                Y[b] = a
        return Y

    def predict(self, X):
        index = [0] * len(X)
        if self.normalizing == True:
            X = self.normalize(X)
        for a in range(len(X)):
            min_distances = 100000
            for b in range(len(self.centroids)):
                distances = self.calculate_distance(self.centroids[b], X[a])
                if distances < min_distances:
                    min_distances = distances
                    index[a] = b
        return index

    def initialize_centroids(self, X, mode):
        if mode == 'random':
            centroids = random.sample(X, self.n_cluster)
        elif mode == 'kmeans++':
            centroids = random.sample(X, 1)
            while(len(centroids) != self.n_cluster):
                dist = [0] * len(X)
                # calculate distance between data and centroids
                for a in range(len(X)):
                    temp = [0] * len(centroids)
                    for b in range(len(centroids)):
                        temp[b] = math.trunc(abs(self.calculate_distance(X[a], centroids[b])))
                    dist[a] = min(temp)
                cummulative = [0] * len(dist)
                cummulative[0] = dist[0]
                # cummulative distance
                for a in range(1, len(dist)):
                    cummulative[a] = dist[a] + cummulative[a-1]
                random_number = random.randint(cummulative[0], cummulative[len(cummulative)-1])
                minimum = 1000000
                # closest random number to distance
                for a in range(len(cummulative)):
                    diff = abs(random_number - cummulative[a])
                    if diff <= minimum:
                        minimum = diff
                        index = a
                    else:
                        break
                centroids.append(X[index])
        return centroids

    def calculate_distance(self, first, second):
        if len(first) != len(second):
            raise 'len tidak sama'
        result = [math.pow(first[a]-second[a], 2) for a in range(len(first))]
        return math.pow(sum(result), 1/2)

    def calculate_new_centroids(self, centroids_members):
        new_centroids = [0] * self.n_cluster
        for a in range(len(new_centroids)):
            new_points = []
            for c in range(len(centroids_members[0][0])):
                new = sum([b[c] for b in centroids_members[a]])/len(centroids_members[a])
                new_points.append(new)
            new_centroids[a] = new_points
        return new_centroids

    def normalize(self, data):
        minim = 0
        maxim = 0
        for a in range(len(data[0])):
            column = [data[b][a] for b in range(len(data))]
            minim = min(column)
            maxim = max(column)
            for b in range(len(data)):
                data[b][a] = (data[b][a]-minim)/(maxim-minim)
        return data

    def sse(self):
        jumlah = 0
        for a in range(len(self.clustered)):
            for b in range(len(self.clustered[a])):
                for c in range(len(self.clustered[a][b])):
                    jumlah += math.pow(self.centroids[a][c] - self.clustered[a][b][c], 2)
        return jumlah