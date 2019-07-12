#import memory_control
import os
import re
from glob import glob
from typing import List

 
prefix = '/spineteam/data/Fertig 20190503/' 


def ct_dirs(prefix):
    return list(glob(prefix + '/*/'))


SINGLE_CT_DIR_LIST: List[str] = [
    ct_dir
    for ct_dir in ct_dirs(prefix)[:1]
]

ALL_CT_DIRS: List[str] = [
    ct_dir
    for ct_dir in ct_dirs(prefix)
    if ct_dir.split('/')[-2][0] != '_'
]

TRAIN_CT_DIRS: List[str] = [
    ct_dir
    for ct_dir in ALL_CT_DIRS if ct_dir.split('/')[-2][0] == '1' 
    or ct_dir.split('/')[-2][0] == '5' 
]

VALIDATION_CT_DIRS: List[str] = [
    ct_dir
    for ct_dir in ALL_CT_DIRS
    if ct_dir not in TRAIN_CT_DIRS
]



def patient_number_from_long_string(ct_dir):
    pat_num = re.findall(r'[\\/](\d\d\d\d)[\\/]', ct_dir)
    if pat_num == []:
        return 0
    else:
        return pat_num[-1]


# check that each patient number only occurs once
for dataset in [TRAIN_CT_DIRS, VALIDATION_CT_DIRS]:
    assert len(dataset) == len(set(dataset))
    assert len(dataset) == len(set(map(patient_number_from_long_string, dataset)))

if __name__ == '__main__':
    print('locations', {prefix: len(list(os.listdir(prefix)))})
    print('all patients', len(ALL_CT_DIRS))
    print('train patients', len(TRAIN_CT_DIRS))
    print('validation patients', len(VALIDATION_CT_DIRS))
    print()

    from load_data import negatives_and_positives_vertebrae

    negatives, positives = negatives_and_positives_vertebrae(VALIDATION_CT_DIRS)
    print('negative validation vertebrae', len(negatives))
    print('positive validation vertebrae', len(positives))
    print()
    negatives, positives = negatives_and_positives_vertebrae(TRAIN_CT_DIRS)
    print('negative train vertebrae', len(negatives))
    print('positive train vertebrae', len(positives))
    print(TRAIN_CT_DIRS.index('/spineteam/data/Fertig 20190503/1003/'))
