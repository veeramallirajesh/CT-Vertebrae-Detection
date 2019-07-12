import numpy

from keras import backend

from load_data.ct_volume import ct_volume
from tuned_cache import TunedMemory

scaled_volume_cache = TunedMemory(cachedir='./.cache/scaled_volume_cache', verbose=0)


# @profile
@scaled_volume_cache.cache
def scaled_ct_volume(shape, image_files, offset=0, shape_mode='scale_smooth', pixel_scaling='range'):
    volume = ct_volume(shape, image_files, offset, shape_mode).astype(backend.floatx())
    assert volume.shape == shape
    if numpy.max(volume) == numpy.min(volume):
        print('WARNING: Unable to scale, returning as is.')
        return volume
    else:
        if pixel_scaling == 'range':
            volume -= numpy.min(volume)  # minimum is 0 now
            volume /= numpy.max(volume)  # maximum is 1 now
            volume -= 0.5  # range is -0.5 to 0.5 now
            volume *= 2  # range is -1 to 1 now
            assert -1 == numpy.min(volume)
            assert numpy.max(volume) == 1
        elif pixel_scaling == 'divide':
            volume /= 256
        else:
            raise NotImplementedError()
        assert volume.shape == shape
        return volume
