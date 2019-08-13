from matplotlib import pyplot as plt
from matplotlib import patches 
import numpy as np
import sys 
import pandas, logging, imageio

#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

output = pandas.read_csv("output{}.csv".format(sys.argv[1]))
indices = np.random.choice(output.shape[0], size=(2, 2))

fig, ax = plt.subplots(2, 2)
for i in range(2):
	for j in range(2):
		index = indices[i][j]
		ax[i][j].imshow(imageio.imread("../Images/images/{}".format(output["image_name"][index])))
		x1, x2, y1, y2 = output["x1"][index], output["x2"][index], output["y1"][index], output["y2"][index]
		x2 = min([640-1,x2])
		y2 = min([480-1,y2])
		print(x1, x2, y1, y2,index)
		rect = patches.Rectangle((x1, y1), (x2 - x1), (y2- y1), linewidth=1, edgecolor='r', facecolor='none')
		ax[i][j].add_patch(rect)
plt.show() 