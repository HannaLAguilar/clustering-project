import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

k = 4
X, y = make_blobs(n_samples=3000,
                  n_features=2,
                  centers=k,
                  cluster_std=0.6,
                  random_state=0)

plt.figure()
sns.scatterplot(x=X[:, 0],
                y=X[:, 1],
                hue=y,
                palette=sns.color_palette("Spectral", k),
                edgecolor='k',
                legend=False)
plt.show()
