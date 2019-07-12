import os
import numpy as np
import pydicom
import pydicom.errors
from PIL import Image
import pandas as pd
from collections import OrderedDict

from tuned_cache import TunedMemory
from util import zoom_to_shape

# @profile
unprocessed_volume_cache = TunedMemory(cachedir='./.cache/unprocessed_volume_cache', verbose=0)

df = pd.read_excel('/spineteam/data/DiagBilanz_Fx_Status_Radiologist_20190604.xlsx', header=1)
df_list = list(OrderedDict.fromkeys(df['Filename']))

@unprocessed_volume_cache.cache
def ct_volume(shape, image_files, offset=0, shape_mode='scale_smooth', additional_offset=False):

    n_split = image_files[0].split(os.path.sep)[:-1]

    if len(image_files) > 1:
        names = [n.split(os.path.sep)[-1] for n in image_files if n.split(os.path.sep)[-1] in df_list]
        if names:
            image_files = [os.path.sep.join([os.path.sep.join(n_split), names[-1]])]
        else:
            image_files = [image_files[0]]


    if shape_mode == 'random_crop':
        print('WARNING: This will be cached to disk and might take up some space')
        volume = np.zeros(shape, dtype=np.int16)
        end = offset + shape[2]

        for idx in range(offset, end):
            if 0 <= idx < len(image_files):
                volume[:, :, idx] = pydicom.dcmread(image_files[idx]).pixel_array
        return volume
    elif shape_mode == 'scale':
        volume = ct_volume(shape='original',
                           image_files=image_files,
                           shape_mode='original')
        return zoom_to_shape(volume, shape, mode='constant')
    elif shape_mode == 'original':
        assert shape == 'original'
        volume = np.asarray(Image.open(image_files[0]))
        volume = volume - [1024]

        return volume
    elif shape_mode == 'scale_smooth':
        volume = ct_volume(shape='original',
                           image_files=image_files,
                           shape_mode='original')
        return zoom_to_shape(volume, shape, mode='smooth')
    else:
        raise NotImplementedError()


