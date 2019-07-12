import os


from load_data import image_files_for_ct
from load_data.ct_metadata import metadata
from train_test_split import patient_number_from_long_string, TRAIN_CT_DIRS, VALIDATION_CT_DIRS

from PIL import Image
import pandas as pd
from collections import OrderedDict

# vertebra_volume_cache = TunedMemory(cachedir='./.cache/vertebra_volume_cache', verbose=0)

dataset = {
    'train': TRAIN_CT_DIRS,
    'test': VALIDATION_CT_DIRS
}



df = pd.read_excel('/spineteam/data/DiagBilanz_Fx_Status_Radiologist_20190604.xlsx', header=1)
df_list = list(OrderedDict.fromkeys(df['Filename']))

out_dir = '/spineteam/data/tiff_patches'
os.makedirs(out_dir, exist_ok=True)



for data in dataset.keys():

    os.makedirs('{0}/{1}/'.format(out_dir, data), exist_ok=True)

    for i in range(len(dataset[data])):

        image_files = image_files_for_ct([dataset[data][i]])[dataset[data][i]]

        n_split = image_files[0].split(os.path.sep)[:-1]

        if len(image_files) > 1:
            names = [n.split(os.path.sep)[-1] for n in image_files if n.split(os.path.sep)[-1] in df_list]
            if names:
                image_files = [os.path.sep.join([os.path.sep.join(n_split), names[-1]])]
            else:
                image_files = [image_files[0]]

        image_file = image_files[0]

        img = Image.open(image_file)
        # np_im = np.asarray(img)


        patient_number = patient_number_from_long_string(image_file)

        image_metadata = metadata(image_file, ignore=['PixelData', 'pixel_array', 'crf', '3dcpm'])
        image_metadata = image_metadata['per_vertebra_annotations']

        for v in image_metadata.keys():
            tiff_metadata = image_metadata[v]['tiff_metadata']
            cen_x_px = (image_metadata[v]['Morphometry Point1X'] +
                                image_metadata[v]['Morphometry Point4X']) / 2
            cen_y_px = (image_metadata[v]['Morphometry Point1Y'] +
                                image_metadata[v]['Morphometry Point4Y']) / 2
            assert isinstance(image_metadata[v]['Flip LR'], int)
            assert isinstance(image_metadata[v]['Flip UD'], int)
            if image_metadata[v]['Flip LR']:
                cen_x_px = (tiff_metadata['ImageWidth'][0] - cen_x_px)
            if image_metadata[v]['Flip UD']:
                cen_y_px = (tiff_metadata['ImageLength'][0] - cen_y_px)

            x_1 = cen_x_px - 60
            y_1 = cen_y_px - 60
            x_2 = cen_x_px + 60
            y_2 = cen_y_px + 60

            box = (x_1, y_1, x_2, y_2)

            cropped_image = img.crop(box)

            if data == 'test':
                cropped_image.save('{0}/{1}/{2}_{3}.tif'.format(out_dir, data, patient_number, v))
            else:
                if image_metadata[v]['SQ Score'] == 0:
                    os.makedirs('{0}/{1}/normal/'.format(out_dir, data), exist_ok=True)
                    cropped_image.save('{0}/{1}/normal/{2}_{3}.tif'.format(out_dir, data, patient_number, v))
                else:
                    os.makedirs('{0}/{1}/fractured/'.format(out_dir, data), exist_ok=True)
                    cropped_image.save('{0}/{1}/fractured/{2}_{3}.tif'.format(out_dir, data, patient_number, v))






