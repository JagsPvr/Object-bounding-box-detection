import matplotlib.pyplot as plt 
from keras.models import load_model
import pandas as pd 
import numpy as np 
import time 
import matplotlib.patches as patches 
MODEL_PATH = './saved_weights/epoch_5/batch_14.h5'
DATA_RESIZED_PATH = './data/X_resized_train_batch_{}.npz'
DATA_ORIG_PATH = './data/X_train_batch_{}.0.npy'
def get_bouding_box_and_iou(y_pred, y_orig):
    """
    Calculate the Intersection over Union of 2 bounding boxes

    Parameters
    ----------
    y_pred : array : [x1, y1, width, height]
    y_orig : array : [x1, y1, width, height]

    Returns
    -------
    float iou in [0., 1.]
    """
    bb1 = {
        'x1': y_pred[0],
        'x2': y_pred[0] + y_pred[2],
        'y1': y_pred[1],
        'y2': y_pred[1] + y_pred[3]
    }
    bb2 = {
        'x1': y_orig[0],
        'x2': y_orig[0] + y_orig[2],
        'y1': y_orig[1],
        'y2': y_orig[1] + y_orig[3]
    }
    print(bb1)
    print(bb2)
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def main():
    Y_train = pd.read_csv('./training')
    Y_train = Y_train.drop('image_name', axis=1)
    Y_train['width'] = Y_train['x2'] - Y_train['x1']
    Y_train['height'] = Y_train['y2'] - Y_train['y1']
    Y_train = Y_train.drop('x2', axis=1)
    Y_train = Y_train.drop('y2', axis=1)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    model = load_model(MODEL_PATH)

    fact = 10
    cols = 8
    fig, ax = plt.subplots(4, cols, figsize=(4 * fact, cols * fact))
    for i in range(cols):
        X_resized_train = np.load(DATA_RESIZED_PATH.format(i + 1))
        X_resized_train = X_resized_train['arr_0']
        X_train = np.load(DATA_ORIG_PATH.format(i + 1))
        y_low = i * 1000
        Y_pred = model.predict(X_resized_train[0:4])
        for j in range(4):
            iou = get_bouding_box_and_iou(Y_pred[j], Y_train.iloc[y_low + j])
            # Showing the original image
            ax[j][i].imshow(X_train[j])
            x1, y1, width, height = Y_train.iloc[y_low + j]
            x1d, y1d, widthd, heightd = Y_pred[j]
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='g', facecolor='none')
            rectd = patches.Rectangle((x1d, y1d), widthd, heightd, linewidth=1, edgecolor='r', facecolor='none')
            ax[j][i].add_patch(rect)
            ax[j][i].add_patch(rectd)
            ax[j][i].set_title('IoU = {}'.format(iou))
            ax[j][i].axis('off')
        del X_resized_train
        del X_train 
    fig.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    main()