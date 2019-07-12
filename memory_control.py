import sys

import tensorflow as tf

# assert 'keras' not in sys.modules

# limit GPU memory usage
gpu_memory_limit = 0.8
print('Limiting GPU memory usage to {0}% per process.'.format(gpu_memory_limit * 100))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_limit
sess = tf.Session(config=config)

# noinspection PyPep8
from keras.backend.tensorflow_backend import set_session

set_session(sess)  # set this TensorFlow session as the default session for Keras

# limit RAM usage
try:
    from memory_limit_windows import create_job, limit_memory, assign_job

    ram_limit = 7 * 1024 * 1024 * 1024  # 7GB
    print('Limiting RAM usage to {0} Bytes.'.format(ram_limit))
    assign_job(create_job())
    limit_memory(ram_limit)
except ModuleNotFoundError:
    print('Setting memory limit failed. Are you on windows?')
