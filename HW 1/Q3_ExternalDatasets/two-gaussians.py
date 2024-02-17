import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eigvals
# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

# Set random seed for reproducibility
np.random.seed(42)



# Generate two multivariate Gaussian random variable objects
mean1, cov1 = [1, 3], [[1, 0.3], [0.3, 1]]
e=eigvals(cov1)
print("Eigenvalues of cov1 are:",e)
rv1 = multivariate_normal(mean1, cov1)

mean2, cov2 = [6, 2], [[1, -0.5], [-0.5, 1]]
e=eigvals(cov2)
print("Eigenvalues of cov2 are:",e)
rv2 = multivariate_normal(mean2, cov2)


data1 = rv1.rvs(size = 100)
data2 = rv2.rvs(size = 100)

# Create a scatter plot
fig	 = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data1[:, 0], data1[:, 1], label='Sea Bass', alpha=0.7,c="blue")
ax.scatter(data2[:, 0], data2[:, 1], label='Salmon', alpha=0.7, c="orange")

x1, x2 = np.mgrid[-1:8:.01, -1:6:.01]
pos = np.dstack((x1, x2))
ax.contour(x1, x2, rv1.pdf(pos),levels=5,colors="blue")
ax.contour(x1, x2, rv2.pdf(pos), levels=5, colors="orange")


ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()


plt.savefig("two-gaussians.pdf", format="pdf", bbox_inches="tight")
