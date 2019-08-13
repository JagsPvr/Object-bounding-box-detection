from keras.optimizers import Adam
from keras.callbacks import Callback
import pandas as pd 
import numpy as np 
from model import Model
import matplotlib.pyplot as plt
Y_train = pd.read_csv('./training')
Y_train = Y_train.drop('image_name', axis=1)
Y_train['width'] = Y_train['x2'] - Y_train['x1']
Y_train['height'] = Y_train['y2'] - Y_train['y1']

Y_train = Y_train.drop('x2', axis=1)
Y_train = Y_train.drop('y2', axis=1)

print("----------------------------------------------------------")
adam = Adam(lr=0.0002)
model = Model()
model = model.get_model()
with open('model_summary.txt', 'w') as model_summary:
    model.summary(print_fn = lambda x: model_summary.write(x + '\n'))
model.compile(optimizer=adam, loss='mse')

print("Model compiled")
print("----------------------------------------------------------")

DATADIR_PREFIX = './data/X_resized_train_batch_'
DATADIR_SUFFIX = '.npz'

MODEL_PREFIX = './saved_weights/batch_'
MODEL_SUFFIX = '.h5'

LOSS_PREFIX = './losses/loss_batch_'
LOSS_SUFFIX = '.png'

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

print("----------------------------------------------------------")
for i in range(14):
    X_train = np.load(DATADIR_PREFIX + str(i + 1) + DATADIR_SUFFIX)
    X_train = X_train['arr_0']
    print("Dataset loaded for batch {}, shape {}".format(i + 1, X_train.shape))
    y_low = i * 1000
    y_high = (i + 1) * 1000
    losshistory = LossHistory()
    hist = model.fit(X_train, Y_train[y_low:y_high], epochs=1, batch_size=16, callbacks=[losshistory])
    print(hist)
    print("Batch {} completed".format(i + 1))
    print("Saving weights...")
    model.save(MODEL_PREFIX + str(i + 1) + MODEL_SUFFIX)
    print("Saving loss diagram...")
    plt.plot(losshistory.losses)
    plt.savefig(LOSS_PREFIX + str(i + 1) + LOSS_SUFFIX)
    Y_pred = model.predict(X_train[0:4])
    with open('training_summary.txt', "a") as ts:
        ts.write("Batch: {}\n".format(i + 1))
        ts.write("Starting Loss: {}\n".format(losshistory.losses[0]))
        ts.write("Ending Loss: {}\n".format(losshistory.losses[-1]))
        ts.write("Predictions : \n {} \n".format(Y_pred))
    del X_train
    print("Batch completed !")
    print("----------------------------------------------------------")