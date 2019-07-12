import math
import os
import random
import sys
from numbers import Number
from typing import List
import logging

import numpy as np
import imageio

import util
from load_data import image_files_for_ct, multiplicative_gaussian_noise, negatives_and_positives_vertebrae
from load_data.ct_metadata import metadata, MissingTiffError
from load_data.spaced_ct_volume import MADER_SPACING
from load_data.vertebra_volume import vertebra_volume
from mixup import apply_mixup
from print_exc_plus import print_exc_plus
from train_test_split import VALIDATION_CT_DIRS, TRAIN_CT_DIRS, patient_number_from_long_string
from util import print_progress_bar, dummy_computation, shorten_name


logger = logging.getLogger('vertebra')

def vertebra_dataset_size(ct_dirs):
    return sum(map(len, negatives_and_positives_vertebrae(ct_dirs)))


class AnnotatedPatchesGenerator():
    def __init__(self,
                 ct_dirs: List[str],
                 batch_size=1,
                 desired_size_mm=(50, 50),
                 random_noise_percent: Number = 0.,
                 random_order=True,
                 weighted=False,
                 pixel_scaling='divide_by_2k',
                 coordinates_from='tiff_2d_center',
                 with_names=False,
                 random_flip_lr=False,
                 random_shift_px=0,
                 include_age=False,
                 include_sex=False,
                 mixup_rate=0):
        from keras import backend
        self.floatx = backend.floatx
        self.ct_dirs = ct_dirs.copy()
        self.image_files_for_ct = image_files_for_ct(self.ct_dirs)

        self.negatives, self.positives = negatives_and_positives_vertebrae(self.ct_dirs)
        self.all_vertebrae = self.positives + self.negatives

        self.random_order = random_order

        if weighted:
            self.weights = {
                True: (len(self.positives) + len(self.negatives)) / (2 * len(self.positives)),
                False: (len(self.positives) + len(self.negatives)) / (2 * len(self.negatives)),
            }
        else:
            self.weights = {True: 1, False: 1}

        self.step_count = 0
        self.batch_size = batch_size
        self.relevant_metadata = np.zeros((self.batch_size, 2), dtype=backend.floatx())
        deformity_label_names = ['Deformity Wedge',
                                 'Deformity Biconcave',
                                 'Deformity Crush']
        score_label_names = ['SQ Score']
        self.label_names = score_label_names + deformity_label_names

        self.y1 = np.zeros((self.batch_size, 1,), dtype=backend.floatx())  # Labels for fractured or not
        self.y2 = np.zeros((self.batch_size, len(self.label_names),),
                           dtype=backend.floatx())  # Labels for score and deformity

        self.label_weights = np.zeros((self.batch_size,), dtype=backend.floatx())

        self.volumes = None
        self.metadata_cache = {}
        self.names = ['' for _ in range(self.batch_size)]
        self.random_shift_px = random_shift_px
        self.random_noise_percent = random_noise_percent
        self.random_flip_lr = random_flip_lr
        self.pixel_scaling = pixel_scaling
        self.coordinates_from = coordinates_from
        self.desired_size_mm = desired_size_mm
        self.with_names = with_names
        self.mixup_rate = mixup_rate
        self.required_size_mm = tuple(size + self.random_shift_px * 2 * spacing
                                      for size, spacing in zip(self.desired_size_mm, MADER_SPACING))
        self.include_age = include_age
        self.include_sex = include_sex

    def __iter__(self):
        return self

    def __next__(self):
        batch_idx = 0
        while batch_idx < self.batch_size:
            if self.random_order:
                if random.random() < 0.5:
                    ct_dir, vertebra = random.choice(self.positives)
                    if (ct_dir, vertebra) not in self.all_vertebrae:  # if we removed it from the dataset already
                        self.positives.remove((ct_dir, vertebra))
                        continue
                else:
                    ct_dir, vertebra = random.choice(self.negatives)
                    if (ct_dir, vertebra) not in self.all_vertebrae:  # if we removed it from the dataset already
                        self.negatives.remove((ct_dir, vertebra))
                        continue
            else:
                ct_dir, vertebra = self.all_vertebrae[
                    (self.step_count * self.batch_size + batch_idx) % len(self.all_vertebrae)]

            image_files = self.image_files_for_ct[ct_dir]
            if (ct_dir, vertebra) not in self.metadata_cache:
                if len(image_files) > 1000:
                    print('(skipping) volume too large: ', ct_dir, vertebra)
                    self.all_vertebrae.remove((ct_dir, vertebra))
                    continue
                one_image_file = image_files[0]
                try:
                    image_metadata = metadata(one_image_file,
                                              ignore=['PixelData', 'pixel_array', 'crf', '3dcpm', 'tiff'])
                except NotImplementedError:  # missing jpeg plugin for example
                    print('(skipping) unable to load metadata for ', ct_dir, vertebra)
                    logger.debug('unable to load metadata for ', ct_dir, vertebra)
                    self.all_vertebrae.remove((ct_dir, vertebra))
                    continue

                if (vertebra not in image_metadata['per_vertebra_annotations']
                        or math.isnan(image_metadata['per_vertebra_annotations'][vertebra]['SQ Score'])):
                    print('(skipping) no annotation available for ', ct_dir, vertebra)
                    logger.debug('no annotation available for ', ct_dir, vertebra)
                    self.all_vertebrae.remove((ct_dir, vertebra))
                    continue

                # if patient_number_from_long_string(ct_dir) != '3009':
                #     print('(skipping) wrong patient ', ct_dir, vertebra)
                #     all_vertebrae.remove((ct_dir, vertebra))
                #     continue

                try:
                    assert image_metadata is not None
                    age = float(image_metadata.get('PatientAge', '000Y')[:3])
                except ValueError:
                    age = 70

                self.metadata_cache[(ct_dir, vertebra)] = {
                    'PatientAge': age,
                    # 'PatientSex': image_metadata['PatientSex'],
                    **{key: image_metadata['per_vertebra_annotations'][vertebra][key]
                       for key in self.label_names},
                }
                for key in self.label_names:
                    assert not math.isnan(float(self.metadata_cache[(ct_dir, vertebra)][key]))

            image_metadata = self.metadata_cache[(ct_dir, vertebra)]

            try:
                volume = vertebra_volume(ct_dir,
                                         vertebra,
                                         desired_size_mm=self.required_size_mm,
                                         pixel_scaling=self.pixel_scaling,
                                         coordinates_from=self.coordinates_from)
                # coordinate order is now UD, FB, LR
                if self.random_flip_lr and random.random() < 0.5:
                    volume = np.flip(volume, axis=2)
                offset = self.random_shift_px
                actual_shift = [random.randint(-self.random_shift_px, self.random_shift_px) + offset
                                for _ in volume.shape]
                volume = volume[
                         # actual_shift[2]:actual_shift[2] + math.ceil(self.desired_size_mm[2] / MADER_SPACING[2]),
                         actual_shift[1]:actual_shift[1] + math.ceil(self.desired_size_mm[1] / MADER_SPACING[1]),
                         actual_shift[0]:actual_shift[0] + math.ceil(self.desired_size_mm[0] / MADER_SPACING[0]),
                         ]  # cut volume
            except MemoryError:
                print('(skipping) memory error for ', ct_dir)
                logger.debug('memory error for ', ct_dir, vertebra)
                self.all_vertebrae.remove((ct_dir, vertebra))
                continue
            except MissingTiffError:
                print('(skipping) tiff not available for ', ct_dir, vertebra)
                logger.debug('tiff not available for ', ct_dir, vertebra)
                self.all_vertebrae.remove((ct_dir, vertebra))
                continue

            if self.random_noise_percent != 0:
                volume = multiplicative_gaussian_noise(volume, self.random_noise_percent)

            # add the data to the batch
            if self.volumes is None:
                self.volumes = np.zeros((self.batch_size, *volume.shape, 1), dtype=self.floatx())
            self.volumes[batch_idx, :, :, 0] = volume

            age = image_metadata['PatientAge'] if self.include_age else 0
            sex = 0 #1 if image_metadata['PatientSex'] == 'M' and self.include_sex else 0

            self.relevant_metadata[batch_idx, :] = [age / 100, sex]

            # the label
            is_fractured = bool(image_metadata['SQ Score'])
            self.y1[batch_idx] = [float(is_fractured)]
            self.y2[batch_idx] = [float(image_metadata[label])
                                  for label in self.label_names]
            self.label_weights[batch_idx] = float(self.weights[is_fractured])
            self.names[batch_idx] = (ct_dir, vertebra)

            batch_idx += 1
        assert self.volumes is not None
        assert self.volumes.shape[0] == self.batch_size
        assert self.relevant_metadata.shape[0] == self.batch_size
        assert self.y1.shape[0] == self.batch_size
        assert self.y2.shape[0] == self.batch_size
        x = [self.volumes, self.relevant_metadata]
        y = [self.y1, self.y2]
        assert not np.isnan(np.sum(self.volumes))
        assert not np.isnan(np.sum(self.y1))
        assert not np.isnan(np.sum(self.y2))
        assert not np.isnan(np.sum(self.relevant_metadata))
        assert not np.isnan(np.sum(self.label_weights))
        batch = [x, y, [self.label_weights, self.label_weights]]
        if self.mixup_rate > 0:
            apply_mixup(batch, self.mixup_rate, self.batch_size)
        self.step_count += 1
        if self.with_names:
            return (*batch, self.names)
        else:
            return tuple(batch)


def main(generate_images=False, show_images=False, augment=0):
    import memory_control
    dummy_computation(memory_control)
    coordinate_sources = [
        # ('tiff_2d_mader_exceptions', '2d_vertebrae_tiff_mader_with_exception'),
        # ('tiff_2d_mader', '2d_vertebrae_tiff_mader'),
        ('tiff_2d_center', '2d_vertebrae_tiff_center'),
        # ('mader', '2d_vertebrae'),
    ]
    pixel_scalings = [
        # 'divide_by_2k',
        # 'range01',
        'range',
    ]
    for coordinates_from, dirname in coordinate_sources:
        for dataset in [VALIDATION_CT_DIRS, TRAIN_CT_DIRS]:
            for scaling in pixel_scalings:
                for desired_size_mm in [(60, 60)]:
                    print('Now doing', coordinates_from, scaling, len(dataset), desired_size_mm)
                    batch_size = 1
                    if augment:
                        noise = 20
                        shift = augment
                    else:
                        noise = 0
                        shift = 0
                    generator = AnnotatedPatchesGenerator(dataset,
                                                          batch_size=batch_size,
                                                          desired_size_mm=desired_size_mm,
                                                          random_order=False,
                                                          weighted=True,
                                                          random_noise_percent=noise,
                                                          pixel_scaling=scaling,
                                                          coordinates_from=coordinates_from,
                                                          with_names=True,
                                                          random_flip_lr=bool(augment),
                                                          random_shift_px=shift)
                    # print(vertebra_dataset_size(dataset))
                    for idx in range(vertebra_dataset_size(dataset)):
                        print_progress_bar(idx, vertebra_dataset_size(dataset))
                        sys.stdout.flush()
                        x, y, w, names = next(generator)
                        if show_images:
                            x0 = x[0]
                            # print(x0.shape)
                            print(y)
                            # print(w)
                            # view_3d_volume(np.swapaxes(x0[0, :, :, :, 0], axis1=0, axis2=2))

                        if generate_images:
                            # image_height = x[0].shape[3]
                            assert batch_size == 1
                            ct_dir, vertebra = names[0]
                            patient_number = patient_number_from_long_string(ct_dir)
                            full_dirname = 'img/generated/{0}'.format(dirname)
                            os.makedirs(full_dirname, exist_ok=True)
                            filename = '{3}/{1}_{2}_{0}_{4}.png'.format(shorten_name(scaling),
                                                                        shorten_name(patient_number),
                                                                        shorten_name(vertebra),
                                                                        full_dirname,
                                                                        shorten_name(desired_size_mm))
                            imageio.imwrite(filename, x[0][0, :, :, 0])
                        print_progress_bar(idx + 1, vertebra_dataset_size(dataset))
                        sys.stdout.flush()
                    print()


if __name__ == '__main__':
    # noinspection PyBroadException
    log_filename = '/home/uokereke/CT_Vertebrae_Detection/logs/basic_file.log'
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=log_filename,
                        filemode='a+')
    try:
        main(generate_images=True,
             show_images=False,
             augment=0)
    except Exception:
        print_exc_plus()
        exit(-1)
    finally:
        frequency = 2000
        duration = 500
        util.beep(frequency, duration)
