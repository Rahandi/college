import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score

from kmeans import KMeans

data = pd.read_csv('seeds.csv', delimiter='\t')
X = data.drop(['h'], axis=1)
X = X.values.tolist()
Y = data['h']
algo = KMeans(3)
Y_pred = algo.fit(X, normalize=False, init='kmeans++')

# score = silhouette_score(X, Y_pred)
score = algo.sse()
print(score)

# scatter = Y_pred
# scatter_x = data['b']
# scatter_y = data['g']
# color = {0: 'red', 1:'blue', 2:'black'}
# fig, ax = plt.subplots()
# for a in range(len(scatter)):
#     ax.scatter(scatter_x[a], scatter_y[a], c=color[scatter[a]], label=scatter[a], s = 50)
# plt.show()