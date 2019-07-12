import os
from typing import Dict, Any

import pandas

from PIL import Image, TiffTags
from joblib import Memory

from train_test_split import ALL_CT_DIRS, patient_number_from_long_string, TRAIN_CT_DIRS
from util import throws,  dummy_computation

metadata_cache = Memory(location='./.cache/metadata', verbose=0)

def excel_data(filename, patient_number, header=0):
    df = excel_panda(filename, header=header)
    return df[df['Patients Name'] == int(patient_number)]


@metadata_cache.cache
def excel_panda(filename, header=0):
    df = pandas.read_excel(filename, sheet_name='Sheet', header=header)
    return df

class MissingTiffError(Exception):
    pass


@throws([ValueError, NotImplementedError])  # If the image could not be read

@metadata_cache.cache
def metadata(image_file: str, ignore=None) -> Dict:
    if ignore is None:
        ignore = ['PixelData', 'pixel_array', 'crf', '3dcpm', 'per_vertebra_annotations']
    patient_number = patient_number_from_long_string(image_file)

    df = excel_panda('/spineteam/data/DiagBilanz_Fx_Status_Radiologist_20190604.xlsx', header=1)
    df = df[df['Patients Name'] == int(patient_number)]
    genant_0 = df[df['SQ Score'] == 0].shape[0]
    genant_1 = df[df['SQ Score'] == 1].shape[0]
    genant_2 = df[df['SQ Score'] == 2].shape[0]
    genant_3 = df[df['SQ Score'] == 3].shape[0]
    is_fractured = genant_1 + genant_2 + genant_3 > 0

    if 'per_vertebra_annotations' not in ignore:
        per_vertebra_annotations = excel_data('/spineteam/data/DiagBilanz_Fx_Status_Radiologist_20190604.xlsx',
                                              patient_number=patient_number,
                                              header=1)
        per_vertebra_annotations = {
            v['Label']: v
            for v in per_vertebra_annotations.to_dict(orient='index').values()
        }

        sum_score = sum(annotation['SQ Score']
                        for annotation in per_vertebra_annotations.values())

        if 'tiff' not in ignore:
            for vertebra, annotation in per_vertebra_annotations.copy().items():
                found = False
                tiff_metadata = None

                filename_to_search = annotation['Filename'].lower()
                for path, directory, filenames in os.walk(os.path.join('/spineteam/data/Fertig 20190503/',
                                                                       str(patient_number))):
                    if len(filenames) > 1:
                        files = [f for f in filenames if f.lower() == filename_to_search]
                        if files:
                            filename = files[0]
                        else:
                            filename = filenames[0]
                    else:
                        filename = filenames[0]
                    # for filename in [f for f in filenames]:# if f.lower() == filename_to_search]:
                    with Image.open(os.path.join(path, filename)) as img:
                        tiff_metadata = {TiffTags.TAGS[key]: img.tag[key] for key in img.tag.keys()}
                    # if found:
                    #     break
                    assert not found
                    found = True
                    #     break

                if not found:
                    raise MissingTiffError('2D tiff not found: ' + filename_to_search)
                assert tiff_metadata is not None
                per_vertebra_annotations[vertebra]['tiff_metadata'] = tiff_metadata

    else:
        per_vertebra_annotations = None
        sum_score = None

    
    return {
        'patient_number': patient_number,
        'is_fractured': is_fractured,
        'per_vertebra_annotations': per_vertebra_annotations,
        'sum_score': sum_score,
        'genant_0': genant_0,
        'genant_1': genant_1,
        'genant_2': genant_2,
        'genant_3': genant_3,
    }


if __name__ == '__main__':
    import memory_control

    dummy_computation(memory_control)
    from load_data import image_files_for_ct

    one_image_file = image_files_for_ct([TRAIN_CT_DIRS[23]])[TRAIN_CT_DIRS[23]][0]
    # print(one_image_file)
    m = metadata(one_image_file, ignore=['PixelData', 'pixel_array', 'crf', '3dcpm'])#, 'tiff'])
    print(m['per_vertebra_annotations']['T6'])
