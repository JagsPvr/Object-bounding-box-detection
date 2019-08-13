import pandas, logging, sys
from model import Model
import numpy as np
from data_loader import Test_data_loader
from hyperparams import Hyperparams

H = Hyperparams()

#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

model = Model().get_model()
model.load_weights("saved_weights/model_epoch_{}.h5".format(sys.argv[1]))
logger.info("Saved weight model_epoch_{} loaded".format(sys.argv[1]))

test_batch_generator = Test_data_loader(H.test_batch_size)

test_csv = pandas.read_csv("../test.csv")
num_test = test_csv["image_name"].shape[0]

predictions = np.ones(shape=(num_test, 4))
logger.info("Empty predictions array initialized of shape : {}".format(predictions.shape))

for i in range(num_test // H.test_batch_size + 1):
	img_batch = test_batch_generator[i]
	logger.info("img_batch shape : {}".format(img_batch.shape))

	x = model.predict_on_batch(img_batch)
	# print(x[1], type(x[1][0]), type(x))

	predictions[i*H.test_batch_size:(i+1)*H.test_batch_size] = x
	logger.info("predictions made on batch : {}, predictions are of shape : {}".format(i, x.shape))

# predictions[:, 0] = predictions[:, 0] *640
# predictions[:, 1] = predictions[:, 1] *640
# predictions[:, 2] = predictions[:, 2] *480
# predictions[:, 3] = predictions[:, 3] *480
predictions[:, 1] = predictions[:, 0] + predictions[:, 1]
predictions[:, 3] = predictions[:, 2] + predictions[:, 3]

test_csv[['x1', 'x2', 'y1', 'y2']] = predictions
test_csv.to_csv("output{}.csv".format(sys.argv[1]), index=False)
logger.info("Test csv written")