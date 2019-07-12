import os

from joblib import Memory

image_files_for_ct_cache = Memory(location='./.cache/image_files_for_ct', verbose=0)


@image_files_for_ct_cache.cache
def image_files_for_ct(ct_dirs):
    images_files_by_ct = {
        ct_dir: [
            os.path.join(ct_dir, file_name)
            for file_name in next(os.walk(ct_dir))[2]
        ]
        for ct_dir in ct_dirs
    }
    return images_files_by_ct
