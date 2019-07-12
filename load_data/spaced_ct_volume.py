import SimpleITK
import numpy

from keras import backend

from load_data.ct_metadata import metadata
from load_data.ct_volume import ct_volume
from tuned_cache import TunedMemory
from util import throws

spaced_volume_cache = TunedMemory(cachedir='./.cache/spaced_volume_cache', verbose=0)

MADER_SPACING = (1, 1)
MADER_INTERPOLATION = 'b_spline'
SWAP_AXES = [(0, 1)]  # To get the format used by O. Mader


# @profile
@spaced_volume_cache.cache
@throws(NotImplementedError)
def spaced_ct_volume(image_files,
                     desired_spacings=MADER_SPACING,  # Spacing used by O. Mader
                     interpolator=MADER_INTERPOLATION,
                     swap_axes=None,
                     pixel_scaling='divide'):
    if swap_axes is None:
        swap_axes = SWAP_AXES  # To get the format used by O. Mader
    volume = ct_volume(shape='original',
                       image_files=image_files,
                       shape_mode='original').astype(backend.floatx())
    if interpolator == 'b_spline':
        interpolator = SimpleITK.sitkBSpline
    elif interpolator == 'nn':
        interpolator = SimpleITK.sitkNearestNeighbor
    elif interpolator == 'linear':
        interpolator = SimpleITK.sitkLinear
    elif interpolator == 'lanczos_windowed_sinc':
        interpolator = SimpleITK.sitkLanczosWindowedSinc
    else:
        raise NotImplementedError('Unknown interpolation method')

    one_image_file = image_files[0]

    image_metadata = metadata(one_image_file)
    # spacings = (*image_metadata['PixelSpacing'], image_metadata['SpacingBetweenSlices'],)  # spacing in mm
    original_shape = volume.shape  # in pixel
    new_size: numpy.ndarray = numpy.ceil(numpy.array([original for original in original_shape])).astype(int)
    for ax1, ax2 in swap_axes:
        volume = numpy.swapaxes(volume, ax1, ax2)
    img_itk: SimpleITK.Image = SimpleITK.GetImageFromArray(volume)
    # img_itk.SetSpacing(spacings)

    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(desired_spacings)

    img_itk = resampler.Execute(img_itk)
    assert img_itk.GetSpacing() == desired_spacings
    volume = SimpleITK.GetArrayFromImage(img_itk)
    if pixel_scaling == 'range':
        volume -= numpy.min(volume)  # minimum is 0 now
        volume /= numpy.max(volume)  # maximum is 1 now
        volume -= 0.5  # range is -0.5 to 0.5 now
        volume *= 2  # range is -1 to 1 now
        assert numpy.min(volume) == -1
        assert numpy.max(volume) == 1
    elif pixel_scaling == 'range01':
        volume -= numpy.min(volume)  # minimum is 0 now
        volume /= numpy.max(volume)  # maximum is 1 now
        assert numpy.min(volume) == 0
        assert numpy.max(volume) == 1
    elif pixel_scaling == 'divide':
        volume /= 256
    elif pixel_scaling == 'divide_by_2k':
        volume /= 2048
    else:
        raise NotImplementedError()
    return volume
