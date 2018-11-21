import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#im = Image.open("images/sky/sky2.jpg")
im = Image.open("images/sea/sea1.jpg")
#im = Image.open("images/city/city24.jpg")
arr = list(im.getdata())
r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))

pca = PCA(n_components=3, whiten=True).fit(arr)
rgb_pca = pca.transform(arr)

cov_mat_rg = np.cov([r, g])
cov_mat_rb = np.cov([r, b])
cov_mat_gb = np.cov([g, b])

print(np.var(rgb_pca[:,0]))
print(np.var(rgb_pca[:,1]))
print(np.var(rgb_pca[:,2]))

cov_mat_pca_rg = np.cov(rgb_pca[:,0].tolist(), rgb_pca[:,1].tolist())
cov_mat_pca_rb = np.cov(rgb_pca[:,0].tolist(), rgb_pca[:,2].tolist())
cov_mat_pca_gb = np.cov(rgb_pca[:,1].tolist(), rgb_pca[:,2].tolist())

print('RG Covariance Matrix:\n', cov_mat_rg)
print('RB Covariance Matrix:\n', cov_mat_rb)
print('GB Covariance Matrix:\n', cov_mat_gb)

print('PCA-RG Covariance Matrix:\n', cov_mat_pca_rg)
print('PCA-RB Covariance Matrix:\n', cov_mat_pca_rb)
print('PCA-GB Covariance Matrix:\n', cov_mat_pca_gb)

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
