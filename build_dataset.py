# USAGE
# python build_dataset.py

# import the necessary packages
from pyimagesearch import config
from imutils import paths
import shutil
import os
import pandas as pd
from collections import OrderedDict
import random
import numpy as np


df = pd.read_excel(config.FILE_NAME)
df_initial_list = list(df['Filename'])
df_list = list(OrderedDict.fromkeys(df['Filename']))
dict = {}

for i, j in enumerate(df_initial_list):
    if j in dict:
        if df['Differential Diagnosis Category'][i] == "Normal":
            continue
        else:
            dict[j] = df['Differential Diagnosis Category'][i]
    else:
        dict[j] = df['Differential Diagnosis Category'][i]

# # loop over the data splits
# for split in (config.TRAIN, config.TEST, config.VAL):
# 	# grab all image paths in the current split
# 	print("[INFO] processing '{} split'...".format(split))
i = 0
p = os.path.sep.join([config.ORIG_INPUT_DATASET])
imagePaths = list(paths.list_images(p))
random.shuffle(imagePaths)

# loop over the image paths
for imagePath in imagePaths:
	# extract class label from the filename
	filename = imagePath.split(os.path.sep)[-1]
	label = config.CLASSES[1 if dict[filename] == "Normal" else 0]

	# construct the path to the output directory
	if i < int(np.ceil(len(df_list) * 0.4)):
    		dirPath = os.path.sep.join([config.BASE_PATH, config.TEST, label])
	else:
    		dirPath = os.path.sep.join([config.BASE_PATH, config.TRAIN, label])

	# if the output directory does not exist, create it
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

	# construct the path to the output image file and copy it
	if filename in df_list:
		p = os.path.sep.join([dirPath, filename])
		shutil.copy2(imagePath, p)
	
	i += 1