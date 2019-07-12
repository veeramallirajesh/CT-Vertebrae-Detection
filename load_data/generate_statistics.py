import memory_control
import os

from matplotlib import pyplot as plt

import util
from load_data import image_files_for_ct, metadata, ct_volume
from print_exc_plus import print_exc_plus
from train_test_split import ALL_CT_DIRS, TRAIN_CT_DIRS, VALIDATION_CT_DIRS, patient_number_from_long_string
from util import dummy_computation

dirs_path = './img/statistics/pixel_distributions/'
if not os.path.exists(dirs_path):
    os.makedirs(dirs_path)

def safe_int(obj, default):
    try:
        return int(obj)
    except ValueError:
        return default


def savefig(filename):
    plt.savefig(filename + '.png')
    # plt.savefig(filename + '.svg')


def generate_statistics():
    datasets = [
        ('all', ALL_CT_DIRS),
        ('train', TRAIN_CT_DIRS),
        ('val', VALIDATION_CT_DIRS),
    ]
    files_by_ct = image_files_for_ct(ALL_CT_DIRS)

    more_metadata = []
    for ct_dir in VALIDATION_CT_DIRS + TRAIN_CT_DIRS:
        patient_number = patient_number_from_long_string(ct_dir)
        filename = './img/statistics/pixel_distributions/{0}'.format(patient_number)
        image_files = files_by_ct[ct_dir]
        more_metadata.append(metadata(image_files[0], ignore=['PixelData', 'pixel_array', '3dcpm', 'crf']))
        if os.path.isfile(filename + '.png'):
            continue
        print('Generating pixel distribution for', ct_dir)
        volume = ct_volume.ct_volume(shape='original',
                                     image_files=image_files,
                                     shape_mode='original')
        plt.hist([volume.flatten()], bins=100)
        plt.legend(['pixel distribution for {0}'.format(patient_number)])
        savefig(filename)
        plt.clf()

    plt.hist([v['tiff_metadata']['XResolution'][0][0] / 1e6
              for m in more_metadata
              for v in m['per_vertebra_annotations'].values()], bins=25)
    plt.legend(['tiff XResolution'])
    plt.xticks(rotation=45)
    savefig('./img/statistics/x_resolution_clean')
    plt.clf()

    plt.hist([v['tiff_metadata']['YResolution'][0][0] / 1e6
              for m in more_metadata
              for v in m['per_vertebra_annotations'].values()], bins=25)
    plt.legend(['tiff YResolution'])
    plt.xticks(rotation=45)
    savefig('./img/statistics/y_resolution_clean')
    plt.clf()

    plt.hist([float(v['SQ Score'])
              for m in more_metadata
              for v in m['per_vertebra_annotations'].values()], bins=25)
    plt.legend(['SQ Score'])
    savefig('./img/statistics/sq_score_clean')
    plt.clf()

    plt.hist([float(v['Deformity Wedge'])
              for m in more_metadata
              for v in m['per_vertebra_annotations'].values()], bins=25)
    plt.legend(['Deformity Wedge'])
    savefig('./img/statistics/deformity_wedge_clean')
    plt.clf()

    plt.hist([float(v['Deformity Biconcave'])
              for m in more_metadata
              for v in m['per_vertebra_annotations'].values()], bins=25)
    plt.legend(['Deformity Biconcave'])
    savefig('./img/statistics/deformity_biconcave_clean')
    plt.clf()

    plt.hist([float(v['Deformity Crush'])
              for m in more_metadata
              for v in m['per_vertebra_annotations'].values()], bins=25)
    plt.legend(['Deformity Crush'])
    savefig('./img/statistics/deformity_crush_clean')
    plt.clf()

    for ds_name, dataset in datasets:
        if len(dataset) == 0:
            print('WARNING: empty dataset: ' + ds_name)
            continue
        print()
        print('Analyzing dataset ' + ds_name)
        # single_image_path = os.path.join(dataset[0], list(os.walk(dataset[0]))[0][2][0])
        # print('Metadata for {0}:'.format(single_image_path))
        # a = pydicom.dcmread(single_image_path)
        # print_attributes(a, ignore=['pixel_array', 'PixelData', 'per_vertebra_annotations'])
        # view_3d_image(images_path)
        # print()
        num_ct_images = len(dataset)
        print('number of ct images:', num_ct_images)
        all_metadata = []
        for ct_dir in dataset:
            assert os.path.isdir(ct_dir)
            image_files = files_by_ct[ct_dir]
            if not image_files:
                raise AssertionError('no images in ' + ct_dir)
            try:
                all_metadata.append(
                    metadata(image_files[0], ignore=['PixelData', '3dcpm', 'per_vertebra_annotations']))
                all_metadata[-1]['Slices'] = len(image_files)
                if len(image_files) > 500:
                    print('{0} has a large number of slices ({1}).'.format(ct_dir, len(image_files)))
            except NotImplementedError:
                print(image_files[0], 'could not be read, might be JPEG-compressed')
                num_ct_images -= 1
                continue
        print('number of readable ct images:', num_ct_images)
        assert len(all_metadata) == num_ct_images
        print('male: ', len([m for m in all_metadata if m['PatientSex'] == 'M']))
        print('not male: ', len([m for m in all_metadata if m['PatientSex'] != 'F']))
        plt.hist([m.get('PatientSex', 'Unknown') for m in all_metadata], bins=25)
        plt.xticks(rotation=90)
        plt.legend(['PatientSex'])
        savefig('./img/statistics/PatientSex_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m.get('PatientPosition', 'Unknown') for m in all_metadata], bins=25)
        plt.xticks(rotation=90)
        plt.legend(['PatientPosition'])
        savefig('./img/statistics/PatientPosition_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m.get('InstitutionName', 'Unknown') for m in all_metadata], bins=25)
        plt.xticks(rotation=90)
        plt.legend(['InstitutionName'])
        savefig('./img/statistics/InstitutionName_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m.get('RotationDirection', 'Unknown') for m in all_metadata], bins=25)
        plt.xticks(rotation=90)
        plt.legend(['RotationDirection'])
        savefig('./img/statistics/RotationDirection_{0}'.format(ds_name))
        plt.clf()
        plt.hist([str(m.get('PatientOrientation', 'Unknown')) for m in all_metadata], bins=25)
        plt.xticks(rotation=90)
        plt.legend(['PatientOrientation'])
        savefig('./img/statistics/PatientOrientation_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m.get('BodyPartExamined', 'Unknown') for m in all_metadata], bins=25)
        plt.legend(['BodyPartExamined'])
        savefig('./img/statistics/BodyPartExamined_{0}'.format(ds_name))
        plt.clf()
        plt.hist([safe_int(m.get('PatientAge', '000Y')[:3], 0) for m in all_metadata], bins=25)
        plt.legend(['PatientAge'])
        savefig('./img/statistics/PatientAge_{0}'.format(ds_name))
        plt.clf()
        plt.hist([safe_int(m.get('PatientAge', '000Y')[:3], 0) for m in all_metadata if m['PatientSex'] == 'M'],
                 bins=25)
        plt.legend(['PatientAge male only'])
        savefig('./img/statistics/PatientAge_Male_{0}'.format(ds_name))
        plt.clf()
        plt.hist([safe_int(m.get('PatientAge', '000Y')[:3], 0) for m in all_metadata if m['PatientSex'] == 'F'],
                 bins=25)
        plt.legend(['PatientAge female only'])
        savefig('./img/statistics/PatientAge_Female_{0}'.format(ds_name))
        plt.clf()
        plt.hist([int(m['KVP']) for m in all_metadata], bins=25)
        plt.legend(['KVP'])
        savefig('./img/statistics/KVP_{0}'.format(ds_name))
        plt.clf()
        plt.hist([int(m['Columns']) for m in all_metadata], bins=25)
        plt.legend(['#columns per slice'])
        savefig('./img/statistics/Columns_{0}'.format(ds_name))
        plt.clf()
        plt.hist([int(m['Rows']) for m in all_metadata], bins=25)
        plt.legend(['#rows per slice'])
        savefig('./img/statistics/Rows_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m['Slices'] for m in all_metadata], bins=25)
        plt.legend(['#slices'])
        savefig('./img/statistics/Slices_{0}'.format(ds_name))
        plt.clf()
        for m in all_metadata:
            assert m['PixelSpacing'][0] == m['PixelSpacing'][1]
        plt.hist([float(m['PixelSpacing'][0]) * float(m['Columns']) for m in all_metadata], bins=25)
        plt.legend(['slice size horizontal (mm)'])
        savefig('./img/statistics/ImageWidth_{0}'.format(ds_name))
        plt.clf()
        plt.hist([float(m['SpacingBetweenSlices']) * float(m['Slices']) for m in all_metadata], bins=25)
        plt.legend(['ct image size vertical (mm)'])
        savefig('./img/statistics/ImageHeight_{0}'.format(ds_name))
        plt.clf()
        plt.hist([float(m['RescaleIntercept']) for m in all_metadata], bins=25)
        plt.legend(['rescale intercept'])
        savefig('./img/statistics/RescaleIntercept_{0}'.format(ds_name))
        plt.clf()
        plt.hist([float(m['RescaleSlope']) for m in all_metadata], bins=25)
        plt.legend(['rescale slope'])
        savefig('./img/statistics/RescaleSlope_{0}'.format(ds_name))
        plt.clf()
        plt.hist([float(m['SliceThickness']) for m in all_metadata], bins=25)
        plt.legend(['slice thickness vertical (mm)'])
        savefig('./img/statistics/SliceThickness_{0}'.format(ds_name))
        plt.clf()
        plt.hist([float(m['SpacingBetweenSlices']) for m in all_metadata], bins=25)
        plt.legend(['vertical spacing between slices (mm)'])
        savefig('./img/statistics/SpacingBetweenSlices_{0}'.format(ds_name))
        plt.clf()
        plt.hist([float(m['SpacingBetweenSlices']) * float(m['Slices']) *
                  float(m['PixelSpacing'][0]) * float(m['Columns']) *
                  float(m['PixelSpacing'][0]) * float(m['Rows'])
                  / 1000000
                  for m in all_metadata], bins=25)
        plt.legend(['total image size (l)'])
        savefig('./img/statistics/ImageLiters_{0}'.format(ds_name))
        plt.clf()
        plt.hist([float(m['Slices']) * float(m['Columns']) * float(m['Rows'])
                  / 1000000
                  for m in all_metadata], bins=25)
        plt.legend(['total image size (Mio. pixel)'])
        savefig('./img/statistics/ImagePixels_{0}'.format(ds_name))
        plt.clf()
        plt.hist([str(m['is_fractured']) for m in all_metadata], bins=25)
        plt.legend(['is fractured'])
        savefig('./img/statistics/is_fractured_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m['genant_0'] for m in all_metadata], bins=25)
        plt.legend(['#vertebra with genant 0'])
        savefig('./img/statistics/genant_0_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m['genant_1'] for m in all_metadata], bins=25)
        plt.legend(['#vertebra with genant 1'])
        savefig('./img/statistics/genant_1_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m['genant_2'] for m in all_metadata], bins=25)
        plt.legend(['#vertebra with genant 2'])
        savefig('./img/statistics/genant_2_{0}'.format(ds_name))
        plt.clf()
        plt.hist([m['genant_3'] for m in all_metadata], bins=25)
        plt.legend(['#vertebra with genant 3'])
        savefig('./img/statistics/genant_3_{0}'.format(ds_name))
        plt.clf()


if __name__ == '__main__':
    dummy_computation(memory_control)
    # noinspection PyBroadException
    try:
        generate_statistics()
    except Exception:
        print_exc_plus()
        exit(-1)
    finally:
        frequency = 2000
        duration = 500
        util.beep(frequency, duration)
