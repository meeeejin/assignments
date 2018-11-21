from scipy import stats
from matplotlib import cm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import bivariate_normal

images = ('sky2', 'sea1', 'city24')

for image in images:
    im = Image.open("images/" + image + ".jpg")
    arr = list(im.getdata())
    r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))

    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    colors = (r, g, b)
    for i, color in enumerate(colors):
        fig = plt.figure()
        ax = plt.subplot(111, projection="3d")

        if i == 0:
            plt.title("Red-Green")
            ax.set_xlabel("Red Intensity")
            ax.set_ylabel("Green Intensity")
            m1 = r
            m2 = g
        elif i == 1:
            plt.title("Red-Blue")
            ax.set_xlabel("Red Intensity")
            ax.set_ylabel("Blue Intensity")
            m1 = r
            m2 = b
        elif i == 2:
            plt.title("Green-Blue")
            ax.set_xlabel("Green Intensity")
            ax.set_ylabel("Blue Intensity")
            m1 = g
            m2 = b
        
        xmin, xmax = m1.min(), m1.max()
        ymin, ymax = m2.min(), m2.max()

        num1 = np.size(m1)
        num2 = np.size(m2)

        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z = np.multiply(Z, 10000)

        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
     
        plt.xticks([0, 50, 100, 150, 200, 250])
        plt.yticks([0, 50, 100, 150, 200, 250])
        
        plt.savefig(image + "_2d_" + str(i) + ".png", dpi=300)

