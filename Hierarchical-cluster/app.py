import numpy as np
import pandas as pd

from hierarchical import Hierarchical

data = pd.read_csv('seeds.csv', delimiter='\t')
X = data.drop(['h'], axis=1)
X = X.values.tolist()
Y = data['h']
algo = Hierarchical(3)
clustered = algo.fit(X)