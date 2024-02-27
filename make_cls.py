import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.datasets import make_blobs

base_path = './artificial_instances'

n = 200
k = 5
sd = 1.0

dataset_name = 'art_' + str(n) + '_' + str(k) + '_' + str(sd)

X, y_true = make_blobs(n_samples=n, centers=k, cluster_std=sd, random_state=0)

plt.title(dataset_name)
plt.scatter(X[:, 0], X[:, 1], s=20)
plt.savefig(base_path + '/plot/' + dataset_name + ".png", bbox_inches='tight')
plt.show()

# create dataset and save it
dataset = pd.DataFrame(X)
n, d = dataset.shape
f = open(base_path + '/data/' + dataset_name + ".txt", 'w')
f.write(str(n) + " " + str(d) + "\n")
f.close()
dataset.to_csv(base_path + '/data/' + dataset_name + ".txt", index=False, header=None, sep=' ', mode="a")
print('Done!')