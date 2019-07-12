import os
import imageio
from math import floor

import numpy

from load_data import image_files_for_ct
from load_data.ct_metadata import metadata
from load_data.spaced_ct_volume import spaced_ct_volume, MADER_INTERPOLATION, MADER_SPACING
from train_test_split import patient_number_from_long_string, ALL_CT_DIRS
from util import throws
from tuned_cache import TunedMemory

vertebra_volume_cache = TunedMemory(cachedir='./.cache/vertebra_volume_cache', verbose=0)


# print('Warning vertebra_volume_cache disabled')
@vertebra_volume_cache.cache
@throws(NotImplementedError)
def vertebra_volume(ct_dir: str,
                    vertebra,
                    desired_size_mm,
                    interpolator=MADER_INTERPOLATION,  # mader uses b_spline
                    pixel_scaling='divide_by_2k',
                    coordinates_from='tiff_2d_center'):
    spacings = list(MADER_SPACING)
    image_files = image_files_for_ct([ct_dir])[ct_dir]

    volume: numpy.ndarray = spaced_ct_volume(image_files,
                                             desired_spacings=MADER_SPACING,
                                             interpolator=interpolator,
                                             pixel_scaling=pixel_scaling)

    volume = numpy.rot90(volume, 3)

    one_image_file = image_files[0]
    patient_number = patient_number_from_long_string(image_files[0])
    if coordinates_from == 'tiff_2d_center':
        image_metadata = metadata(one_image_file, ignore=['PixelData', 'pixel_array', 'crf', '3dcpm'])
        tiff_metadata = image_metadata['per_vertebra_annotations'][vertebra]['tiff_metadata']
        tiff_center_x_px = (image_metadata['per_vertebra_annotations'][vertebra]['Morphometry Point1X'] +
                            image_metadata['per_vertebra_annotations'][vertebra]['Morphometry Point4X']) / 2
        tiff_center_y_px = (image_metadata['per_vertebra_annotations'][vertebra]['Morphometry Point1Y'] +
                            image_metadata['per_vertebra_annotations'][vertebra]['Morphometry Point4Y']) / 2
        assert isinstance(image_metadata['per_vertebra_annotations'][vertebra]['Flip LR'], int)
        assert isinstance(image_metadata['per_vertebra_annotations'][vertebra]['Flip UD'], int)
        if image_metadata['per_vertebra_annotations'][vertebra]['Flip LR']:
            tiff_center_x_px = (image_metadata['per_vertebra_annotations'][vertebra]['tiff_metadata']['ImageWidth'][0]
                                - tiff_center_x_px)
        if image_metadata['per_vertebra_annotations'][vertebra]['Flip UD']:
            tiff_center_y_px = (image_metadata['per_vertebra_annotations'][vertebra]['tiff_metadata']['ImageLength'][0]
                                - tiff_center_y_px)
        center_px = (
            # volume.shape[2] // 2,
            round(tiff_center_x_px / (tiff_metadata['XResolution'][0][0] / 1e6) / spacings[0]),
            round(tiff_center_y_px / (tiff_metadata['YResolution'][0][0] / 1e6) / spacings[1]),
        )

    else:
        raise NotImplementedError()

    # from PIL import Image
    # import pandas as pd
    # import numpy as np
    # file_loc = "/Users/kavya/Documents/MasterProject/3D_Vertebrae_detection/DiagBilanz_Fx_Status_Radiologist_20190604.xlsx"
    # data_loc = "/Users/kavya/Documents/MasterProject/3D_Vertebrae_detection/Fertig 20190503"
    # cropped_loc = "/Users/kavya/Documents/MasterProject/3D_Vertebrae_detection/cropped"
    # im = Image.open(data_loc + "/1001/1001_UKSH_KIEL_RADIOLOGIE_NEURORAD_KVP80_cExp129.399_PixSp0-1_20Transversals.tif",
    #                 mode='r')
    # # im.show()
    # np_im = np.array(im)
    #
    # df = pd.read_excel(file_loc, header=1, usecols="B,AJ,AP,AX,AZ,BA,BW,BX,CC,CD")
    # print(df.head())
    # print(df.shape)
    # print(df['Patients Name'].iloc[0])
    # print(df['Filename'].iloc[0])
    #
    # SHAPE = 512
    #
    # for i in range(df.shape[0]):
    #     if (df['Patients Name'].iloc[i] == 1008):
    #         continue
    #     file_fetched = df['Filename'].iloc[i]
    #     file_index = file_fetched.rfind(str(df['Patients Name'].iloc[i]))
    #     file_standard = file_fetched[file_index:]
    #     # print(file_standard)
    #     im = Image.open(data_loc + '/' + str(df['Patients Name'].iloc[i]) + '/' + file_standard, mode='r')
    #     # im.show()
    #     Point2X = (SHAPE - df['Morphometry Point2X'].iloc[i]) - 4
    #     Point2Y = (df['Morphometry Point2Y'].iloc[i]) - 4
    #     Point5X = (SHAPE - df['Morphometry Point5X'].iloc[i]) + 4
    #     Point5Y = (df['Morphometry Point5Y'].iloc[i]) + 4
    #     box = (Point2X, Point2Y, Point5X, Point5Y)
    #     print(box)
    #     cropped_image = im.crop(box)
    #     # cropped_image.show()
    #     # crop_im = np.array(cropped_image)
    #     # print(cropped_image)
    #     if ((df['SQ Score'].iloc[i]) == 0):
    #         cropped_image.save(
    #             cropped_loc + '/' + 'Healthy' + '/' + str(df['Patients Name'].iloc[i]) + '_' + df['Label'].iloc[
    #                 i] + '.tif')
    #     else:
    #         cropped_image.save(
    #             cropped_loc + '/' + 'Fracture' + '/' + str(df['Patients Name'].iloc[i]) + '_' + df['Label'].iloc[
    #                 i] + '.tif')

    # mader uses reverse coordinate order from ours
    # desired_size_mm = desired_size_mm[::-1]
    # spacings = spacings[::-1]
    # center_px = center_px[::-1]


    for center, current_length in zip(center_px, volume.shape):
        assert 0 <= center < current_length

    desired_size_px = tuple(round(size / spacing)
                            for size, spacing
                            in zip(desired_size_mm, spacings))

    def _clip(x, minimum, maximum):
        return max(minimum, min(x, maximum))

    # patch around center
    slices = [
        slice(_clip(center - floor(desired / 2), minimum=0, maximum=current_length - desired),
              _clip(center - floor(desired / 2), minimum=0, maximum=current_length - desired) + desired, )
        for center, desired, current_length
        in zip(center_px, desired_size_px, volume.shape)
    ]





    filename = 'img/generated/full_spine_center/{0}.png'.format(patient_number)
    if not os.path.isfile(filename):
        os.makedirs('img/generated/full_spine_center/', exist_ok=True)
        imageio.imwrite(filename, volume[:, :])

    # print("Initial volume shape {}".format(volume.shape))



    volume = volume[slices[0], slices[1]]
    # view_3d_volume(numpy.swapaxes(volume[slices[0], slices[1], slices[2]], axis1=0, axis2=2))
    # view_3d_volume(numpy.swapaxes(volume, axis1=0, axis2=2))

    # filename = 'img/generated/diff/{0}{1}.png'.format(patient_number, vertebra)
    # if not os.path.isfile(filename):
    #     os.makedirs('img/generated/diff/', exist_ok=True)
    #     imageio.imwrite(filename, volume[:, :])

    # print("Slice volume shape {}".format(volume.shape))


    # if image_metadata['PatientPosition'] == 'FFS':
    #     print('FFS in', patient_number)

    assert volume.shape == desired_size_px
    return volume

# vertebra_volume(ALL_CT_DIRS[15], 'T4',(50, 50), coordinates_from='tiff_2d_center')