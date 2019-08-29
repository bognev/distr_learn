import mnist
import scipy.misc
from PIL import Image
import numpy as np

images = mnist.train_images()
img = Image.fromarray(images[0,:,:]* -1 + 256)
img.save('my.png')
img.show()
# scipy.misc.toimage(scipy.misc.imresize(images[0,:,:] * -1 + 256, 10.))
print("finished")