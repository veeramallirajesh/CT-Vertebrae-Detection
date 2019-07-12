import numpy


def shift(x, offset=-1):
    return numpy.concatenate([x[offset:, ...], x[:offset, ...]], axis=0)


def apply_mixup_to_array(x: numpy.ndarray, lam: numpy.ndarray):
    assert lam.shape == (x.shape[0],)
    for _ in x.shape[1:]:
        lam = numpy.expand_dims(lam, axis=-1)
    return lam * x + (1 - lam) * shift(x)


def apply_mixup(batch, alpha, batch_size=None):
    """
    :param batch: a mutable sequence of inputs, labels, label_weights, ...
    :param alpha: the parameter of the beta distribution
    :param batch_size: OPTIONAL the batch size of the batch
    :return: same shape as data
    """
    lam = None
    if batch_size is not None:
        lam = numpy.random.beta(alpha, alpha, batch_size)
    for i in range(len(batch)):
        x = batch[i]
        if isinstance(x, numpy.ndarray):
            if batch_size is None:
                batch_size = x.shape[0]
                lam = numpy.random.beta(alpha, alpha, batch_size)
                lam = numpy.abs(lam - 0.5)
            else:
                assert x.shape[0] == batch_size
            assert batch_size is not None
            assert lam is not None
            batch[i] = apply_mixup_to_array(x, lam)
        else:
            for j in range(len(x)):
                x2 = x[j]
                if batch_size is None:
                    batch_size = x2.shape[0]
                    lam = numpy.random.beta(alpha, alpha, batch_size)
                else:
                    assert x2.shape[0] == batch_size
                assert batch_size is not None
                assert lam is not None
                x[j] = apply_mixup_to_array(x2, lam)


# def mixup(alpha):
#     def mixup_generator(generator=None):
#         @functools.wraps(generator)
#         def wrap(*args, **kwargs):
#             g = generator(*args, **kwargs)
#             data = next(generator)  # data is a tuple of inputs, labels, label_weights, ...
#             if isinstance(data[0], numpy.ndarray):
#                 batch_size = data[0].shape[0]
#             else:
#                 batch_size = data[0][0].shape[0]
#
#             np.random.beta(self.alpha, self.alpha, batch_size)
#             # TODO sample from beta distribution
#             for x in data:
#                 if isinstance(x, numpy.ndarray):
#                     batch_size = x.shape[0]
#                 else:
#                     batch_size = x[0].shape[0]
#
#         if generator is None:
#             return wrap
#         else:
#             return


# GaussianNoise
#
# class Mixup(Layer):
#
#     def __init__(self, mixup, **kwargs):
#         self.mixup = mixup
#         super(Mixup, self).__init__(**kwargs)
#
#     def call(self, inputs, training=None, **kwargs):
#         def mixed():
#             mixup = 1.0 * self.mixup  # Convert to float, as tf.distributions.Beta requires floats.
#             beta = tf.distributions.Beta(mixup, mixup)
#             lam = beta.sample(K.shape(inputs)[0])
#             ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
#
#             def circular_shift(values):  # Circular shift in batch dimension
#                 return tf.concat([values[-1:, ...], values[:-1, ...]], 0)
#
#             transformed = ll * inputs + (1 - ll) * circular_shift(inputs)
#             labels = lam * labels + (1 - lam) * circular_shift(labels)
#             return transformed
#         return K.in_train_phase(mixed, inputs, training=training)
#
#
#
#         assert isinstance(x, list)
#         a, b = x
#         return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]
#
#     def get_config(self):
#         config = {'mixup': self.mixup}
#         base_config = super(Mixup, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape

if __name__ == '__main__':
    x = [numpy.array([[1, 2, 3], [4, 5, 6]]), [numpy.array([0, 1]), numpy.array([0, 1])]]
    print(x)
    apply_mixup(x, 0.2)
    print(x)
