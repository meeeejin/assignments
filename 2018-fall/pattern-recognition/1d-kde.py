from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

bandwidths = (1, 2, 4, 8)
#images = ('sky2', 'sea1', 'city24')
images = ('city24', 'city24')

for image in images:
    im = Image.open("images/" + image + ".jpg")
    arr = list(im.getdata())
    r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))

    X_plot = np.linspace(0, 255, 1000)[:, None]

    r = np.array(r).reshape(-1, 1)
    g = np.array(g).reshape(-1, 1)
    b = np.array(b).reshape(-1, 1)

    colors = (r, g, b)
    for i, color in enumerate(colors):
        fig = plt.figure()
        ax = plt.subplot(111)

        if i == 0:
            plt.title("Red")
        elif i == 1:
            plt.title("Green")
        elif i == 2:
            plt.title("Blue")

        ax.set_xlabel("Intensity")

        for bw in bandwidths: 
            kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(color)
            log_dens = kde.score_samples(X_plot)
            ax.plot(X_plot[:, 0], np.exp(log_dens),
                    label='$Bandwidth = %i$' % bw)
        
            ax.legend(loc='upper right')
        plt.savefig(image + "_1d_" + str(i) + ".png", dpi=300)

