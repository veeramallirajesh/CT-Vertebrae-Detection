# USAGE
# python build_dataset.py

# import the necessary packages
from pyimagesearch import config
from imutils import paths
import shutil
import os
import pandas as pd
from collections import OrderedDict


df = pd.read_excel(config.FILE_NAME)
df_list = list(OrderedDict.fromkeys(df['Filename']))

# # loop over the data splits
# for split in (config.TRAIN, config.TEST, config.VAL):
# 	# grab all image paths in the current split
# 	print("[INFO] processing '{} split'...".format(split))
p = os.path.sep.join([config.ORIG_INPUT_DATASET])
imagePaths = list(paths.list_images(p))

# loop over the image paths
for imagePath in imagePaths:
	# extract class label from the filename
	filename = imagePath.split(os.path.sep)[-1]
	#label = config.CLASSES[int(filename.split("_")[0])]

	# construct the path to the output directory
	dirPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])

	# if the output directory does not exist, create it
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

	# construct the path to the output image file and copy it
	if filename in df_list:
		p = os.path.sep.join([dirPath, filename])
		shutil.copy2(imagePath, p)