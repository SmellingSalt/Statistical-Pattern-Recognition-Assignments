# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# from matplotlib import cm

# fig = plt.figure()
# ax = fig.gca(projection='rectilinear')
# X, Y, Z = axes3d.get_test_data(0.05)
# # ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# # cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# # cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

# ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
# # ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)

# plt.show()

import numpy as np
import pylab as pl
from sklearn import mixture

np.random.seed(0)
C1 = np.array([[1, -0.5],[-0.5, 1]])
C2 = np.array([[1, 0.9],[0.9, 1]])

X_train = np.r_[
    np.random.multivariate_normal((0,10), C1, size=100),
    np.random.multivariate_normal((0, 0), C2, size=100),
]

clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

#define g1(x, y) and g2(x, y)

def g1(x, y):
    return clf.predict_proba(np.column_stack((x, y)))[:, 0]

def g2(x, y):
    return clf.predict_proba(np.column_stack((x, y)))[:, 1]

#plot code from here

X, Y = np.mgrid[-15:15:100j, -15:15:100j]
x = X.ravel()
y = Y.ravel()

p = (g1(x, y) - g2(x, y)).reshape(X.shape)

pl.scatter(X_train[:, 0], X_train[:, 1])
pl.contour(Y, X, p, levels=[0])
