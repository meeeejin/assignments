import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


im = Image.open("images/sky/sky2.jpg")
#im = Image.open("images/city/city24.jpg")
#im = Image.open("images/sea/sea1.jpg")
#r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))

arr = list(im.getdata())
pca = PCA(n_components=3, whiten=True).fit(arr)
rgb_pca = pca.transform(arr)
plt.scatter(rgb_pca[:,0], rgb_pca[:,1], marker='o', color='red', alpha=0.1)
#plt.scatter(rgb_pca[:,0], rgb_pca[:,2], marker='o', color='red', alpha=0.1)
#plt.scatter(rgb_pca[:,1], rgb_pca[:,2], marker='o', color='red', alpha=0.1)

#plt.scatter(g, b, marker='o', color='blue', alpha=0.1)

plt.xlabel("Red")
plt.ylabel("Green")
#plt.xlabel("Green")
#plt.ylabel("Blue")

#plt.savefig("city24-2d-pca-rg.png", dpi=300)
#plt.savefig("city24-2d-pca-rb.png", dpi=300)
#plt.savefig("city24-2d-pca-gb.png", dpi=300)

#plt.savefig("sea1-2d-pca-rg.png", dpi=300)
#plt.savefig("sea1-2d-pca-rb.png", dpi=300)
#plt.savefig("sea1-2d-pca-gb.png", dpi=300)

plt.savefig("sky2-2d-pca-rg.png", dpi=300)
#plt.savefig("sky2-2d-pca-rb.png", dpi=300)
#plt.savefig("sky2-2d-pca-gb.png", dpi=300)


'''
def histogram():
    images = ('city', 'sky', 'sea')
    for j, image in enumerate(images):
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        for k in range(1, 26):
            im = Image.open("images/" + image + "/" + image + str(k) + ".jpg")
            r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))
            axis.scatter(r, g, b, marker='o', color='blue', alpha=0.1)
            axis.set_xlabel("Red")
            axis.set_ylabel("Green")
            axis.set_zlabel("Blue")

        plt.savefig(image + ".png", dpi=300)

histogram()
'''
