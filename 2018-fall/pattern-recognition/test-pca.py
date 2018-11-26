import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

im = Image.open("image.jpg")
arr = list(im.getdata())

pca = PCA(n_components=3, whiten=True).fit(arr)
rgb_pca = pca.transform(arr)

axis.scatter(rgb_pca[:,0], rgb_pca[:,1], rgb_pca[:,2], marker='o', color='red', alpha=0.1, label='PCA')

axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")

plt.savefig("sky2-rgb-pca.png", dpi=300)
