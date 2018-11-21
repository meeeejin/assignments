import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

im = Image.open("images/sky/sky2.jpg")
arr = list(im.getdata())
#r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))

'''
cov_mat = np.cov([r, g, b])
print('Covariance Matrix:\n', cov_mat)
'''
pca = PCA(n_components=3, whiten=True).fit(arr)
rgb_pca = pca.transform(arr)

#axis.scatter(r, g, b, marker='o', color='blue', alpha=0.1, label='Original')
axis.scatter(rgb_pca[:,0], rgb_pca[:,1], rgb_pca[:,2], marker='o', color='red', alpha=0.1, label='PCA')

axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")

#plt.legend(loc='upper right')

plt.savefig("sky2-rgb-pca.png", dpi=300)

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
