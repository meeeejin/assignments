import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#plt.scatter(rgb_pca[:,0], rgb_pca[:,1], marker='o', color='red', alpha=0.1)
#plt.savefig("test.png", dpi=300)

images = ('city', 'sky', 'sea')

features = []
labels = []

for j, image in enumerate(images):
    for k in range(1, 21):
        im = Image.open("images/" + image + "/" + image + str(k) + ".jpg")
        arr = list(im.getdata())
        
        pca = PCA(n_components=2, whiten=True).fit(arr)
        rgb_pca = pca.transform(arr)

        features.append(rgb_pca.flatten())
        labels.append(image)

trainFeat = np.array(features)
trainLabels = np.array(labels)

features = []
labels = []

for j, image in enumerate(images):
    for k in range(21, 26):
        im = Image.open("images/" + image + "/" + image + str(k) + ".jpg")
        arr = list(im.getdata())
        
        pca = PCA(n_components=2, whiten=True).fit(arr)
        rgb_pca = pca.transform(arr)

        features.append(rgb_pca.flatten())
        labels.append(image)

testFeat = np.array(features)
testLabels = np.array(labels)

print("[INFO] evaluating accuracy...")
clf = SVC()
clf.fit(trainFeat, trainLabels)
acc = clf.score(testFeat, testLabels)
predict = clf.predict(trainFeat)
predict2 = clf.predict(testFeat)

print(predict)
print(predict2)
print("[INFO] test accuracy: {:.3f}%".format(acc * 100))
