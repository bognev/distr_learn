import mnist
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
img = Image.fromarray(train_images[0,:,:]* -1 + 256)
# img.save('my.png')
# img.show()
print(train_labels[0])
train_images_r = train_images.reshape(60000,28*28)
test_images_r = test_images.reshape(10000,28*28)
logistic = LogisticRegression(n_jobs=-1)
clf = logistic.fit(X=train_images_r[0:1000], y=train_labels[0:1000]).score(test_images_r[0:100], test_labels[0:100])
print(logistic.predict(test_images_r[1001:10010]))
print(test_labels[1001:10010])
print(clf)

print("finished")