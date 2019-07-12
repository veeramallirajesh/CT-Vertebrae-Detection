import math

from joblib import Memory

from typing import List, Tuple

import numpy as np
# import matplotlib.pyplot as plt

from load_data.ct_metadata import metadata

from load_data.image_files_for_ct import image_files_for_ct
# from multi_slice_viewer import multi_slice_viewer

WHOLE_IMAGE_SHAPE = (512, 512)
VERTEBRAE = ['T' + str(i + 1) for i in range(12)] + ['L' + str(i + 1) for i in range(5)]

positives_and_negatives_cache = Memory(location='./.cache/positives_and_negatives', verbose=0)



# def view_3d_volume(volume: np.ndarray):
#     multi_slice_viewer(volume)
#     plt.show()


@positives_and_negatives_cache.cache
def negatives_and_positives(ct_dirs):
    images_files_by_ct = image_files_for_ct(ct_dirs)
    positives = []
    negatives = []
    for ct_dir in ct_dirs:
        try:
            image_metadata = metadata(images_files_by_ct[ct_dir][0])
        except NotImplementedError:  # missing jpeg plugin for example
            continue
        if image_metadata['is_fractured'] is None:  # No annotation available :(
            raise AssertionError('This should not happen anymore after the metadata rewrite')
        if image_metadata['is_fractured']:
            positives.append(ct_dir)
        else:
            negatives.append(ct_dir)
    return negatives, positives


@positives_and_negatives_cache.cache
def negative_and_positive_scores(ct_dirs):
    images_files_by_ct = image_files_for_ct(ct_dirs)
    positives = []
    negatives = []
    for ct_dir in ct_dirs:
        try:
            image_metadata = metadata(images_files_by_ct[ct_dir][0],
                                      ignore=['PixelData', 'pixel_array', 'crf', '3dcpm', 'tiff'])
        except NotImplementedError:  # missing jpeg plugin for example
            continue
        if math.isnan(image_metadata['sum_score']):
            continue
        if image_metadata['sum_score'] > 0:
            positives.append(ct_dir)
        else:
            negatives.append(ct_dir)
    return negatives, positives


@positives_and_negatives_cache.cache
def negatives_and_positives_vertebrae(ct_dirs: List[str], ignore_missing_annotations=True):
    images_files_by_ct = image_files_for_ct(ct_dirs)
    positives = []
    negatives = []
    for ct_dir in ct_dirs:
        for vertebra in VERTEBRAE:
            try:
                image_metadata = metadata(images_files_by_ct[ct_dir][0],
                                          ignore=['PixelData', 'pixel_array', 'crf', '3dcpm', 'tiff'])
            except NotImplementedError:  # missing jpeg plugin for example
                continue
            if ignore_missing_annotations and \
                    (vertebra not in image_metadata['per_vertebra_annotations']
                     or math.isnan(image_metadata['per_vertebra_annotations'][vertebra]['SQ Score'])):
                # No annotation available :(
                continue

            score = float(image_metadata['per_vertebra_annotations'][vertebra]['SQ Score'])
            if score == 0.:
                negatives.append((ct_dir, vertebra))
            else:
                positives.append((ct_dir, vertebra))
    return negatives, positives



def gaussian_noise(image, std):
    return np.random.normal(image, std)


def multiplicative_gaussian_noise(image, avg_percent):
    std = avg_percent / 100 / math.sqrt(2 / math.pi)
    return image * np.random.normal(1, std)

