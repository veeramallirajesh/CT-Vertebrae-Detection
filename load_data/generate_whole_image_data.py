import os
import random
from typing import List

import numpy as np
from keras import backend

from load_data import gaussian_noise, negatives_and_positives
from load_data.ct_metadata import metadata
from load_data.random_rotate import random_rotate
from load_data.scaled_ct_volume import scaled_ct_volume


def generate_whole_image_data(ct_dirs: List[str],
                              batch_size,
                              desired_shape,
                              random_noise_std=0,
                              random_order=True,
                              verbose=True,
                              weighted=False,
                              rotate_mode='random_v2',
                              shape_mode='scale_smooth',
                              pixel_scaling='range'):
    ct_dirs = ct_dirs.copy()  # we might modify this list later
    images_files_by_ct = {
        ct_dir: [
            os.path.join(ct_dir, file_name)
            for file_name in next(os.walk(ct_dir))[2]
        ]
        for ct_dir in ct_dirs
    }

    positives = []
    negatives = []
    for ct_dir in ct_dirs:
        try:
            image_metadata = metadata(images_files_by_ct[ct_dir][0])
        except NotImplementedError:  # missing jpeg plugin for example
            continue
        if image_metadata['is_fractured'] is None:  # No annotation available :(
            raise AssertionError('This should not be possible after metadata rework')
        if image_metadata['is_fractured']:
            positives.append(ct_dir)
        else:
            negatives.append(ct_dir)
    negatives, positives = negatives_and_positives(ct_dirs)

    if weighted:
        weights = {
            True: (len(positives) + len(negatives)) / (2 * len(positives)),
            False: (len(positives) + len(negatives)) / (2 * len(negatives)),
        }
    else:
        weights = {True: 1, False: 1}

    step_count = 0
    volumes = np.zeros((batch_size, *desired_shape, 1), dtype=backend.floatx())
    relevant_metadata = np.zeros((batch_size, 2), dtype=backend.floatx())

    y = np.zeros((batch_size, 2,), dtype=backend.floatx())  # Categorical labels even for 2-class problem

    label_weights = np.zeros((batch_size,), dtype=backend.floatx())
    while 1:
        batch_idx = 0
        while batch_idx < batch_size:
            if random_order:
                if random.random() < 0.5:
                    ct_dir = random.choice(positives)
                    if ct_dir not in ct_dirs:  # if we removed it from the dataset already
                        positives.remove(ct_dir)
                        continue
                else:
                    ct_dir = random.choice(negatives)
                    if ct_dir not in ct_dirs:  # if we removed it from the dataset already
                        negatives.remove(ct_dir)
                        continue
            else:
                ct_dir = ct_dirs[(step_count * batch_size + batch_idx) % len(ct_dirs)]

            image_files = images_files_by_ct[ct_dir]
            if len(image_files) > 1000:
                if verbose:
                    print('(skipping) volume too large: ', ct_dir)
                continue
            one_image_file = image_files[0]
            try:
                image_metadata = metadata(one_image_file)
            except NotImplementedError:  # missing jpeg plugin for example
                if verbose:
                    print('(skipping) unable to load metadata for ', ct_dir)
                continue
            if image_metadata['is_fractured'] is None:  # No annotation available :(
                if verbose:
                    print('(skipping) no annotation available for ', ct_dir)
                ct_dirs.remove(ct_dir)
                continue

            volume = scaled_ct_volume(desired_shape,
                                      image_files,
                                      offset=0,
                                      shape_mode=shape_mode,
                                      pixel_scaling=pixel_scaling)

            if random_noise_std != 0:
                volume = gaussian_noise(volume, random_noise_std)
            # volume = random_contrast_modifications(volume, random_noise_std)
            # TODO (later) augment whole image
            if rotate_mode == 'random_v2' or rotate_mode == 'random':
                volume = random_rotate(volume, rotation_directions={0, 1, 2})
            elif rotate_mode == 'not':
                pass
            else:
                raise NotImplementedError('Invalid rotate_mode')
            # TODO (later) augment patch
            # patch = random_rotate(patch)
            # add the data to the batch
            volumes[batch_idx, :, :, :] = np.expand_dims(volume, axis=3)

            try:
                age = float(image_metadata.get('PatientAge', '000Y')[:3])
            except ValueError:
                age = 70
            sex = 1 if image_metadata['PatientSex'] == 'M' else 0

            relevant_metadata[batch_idx, :] = [age / 10, sex]

            # the label
            label = float(image_metadata['is_fractured'])
            y[batch_idx] = [1 - label, label]

            label_weights[batch_idx] = float(weights[image_metadata['is_fractured']])
            batch_idx += 1
        assert volumes is not None
        assert volumes.shape[0] == batch_size
        assert relevant_metadata.shape[0] == batch_size
        assert y.shape[0] == batch_size
        x = [volumes, relevant_metadata]
        yield x, y, label_weights
        step_count += 1
