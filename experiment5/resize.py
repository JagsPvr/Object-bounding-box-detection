import numpy as np 
from skimage.transform import resize

ORIG_PREFIX = './data/X_train_batch_'
ORIG_SUFFIX = '.0.npy'

RESIZE_PREFIX = './data/X_resized_train_batch_'
RESIZE_SUFFIX = '.npz'
for i in range(1, 15):
    X_resized_train = []
    X_train = np.load(ORIG_PREFIX + str(i) + ORIG_SUFFIX)
    for img in X_train:
        img_resized = resize(img, (192, 256))
        X_resized_train.append(img_resized)
    X_resized_train = np.asarray(X_resized_train)
    print("Resized shape for batch {} is {}".format(i, X_resized_train.shape))
    np.savez_compressed(RESIZE_PREFIX + str(i) + RESIZE_SUFFIX, X_resized_train)
    del X_train
    del X_resized_train
    print("Iteration complete")
    print("-------------------------------------------------------------")