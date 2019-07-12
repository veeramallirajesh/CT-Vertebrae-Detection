import os
from typing import List, Dict
from joblib import Memory
from load_data.ct_metadata import metadata
from train_test_split import ALL_CT_DIRS, TRAIN_CT_DIRS, VALIDATION_CT_DIRS

label_cache = Memory(location='./.cache/all_labels', verbose=0)


@label_cache.cache
def all_labels(dataset: List[str]) -> Dict[bool, int]:
    result = {label: 0 for label in [True, False, None]}
    for ct_dir in dataset:
        image_files = [
            os.path.join(ct_dir, file_name)
            for file_name in next(os.walk(ct_dir))[2]
        ]
        try:
            is_fractured = metadata(image_files[0])['is_fractured']
        except NotImplementedError:  # Missing JPEG plugin
            is_fractured = None
        result[is_fractured] += 1
    return result


if __name__ == '__main__':
    print(all_labels(ALL_CT_DIRS))
    print(all_labels(TRAIN_CT_DIRS))
    print(all_labels(VALIDATION_CT_DIRS))
