import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow import keras
import tree


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs, **kwargs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    def get_config(self):
        config = super(ResidualUnit, self).get_config()
        config.update({"filters": self.filters,
                       "strides": self.strides,
                       "activation": keras.activations.serialize(self.activation)})
        return config


def custom_resnet_model(inputs):
    """
    Makes a ResNet with 33 layers
    Args:
        inputs: keras.layers.Input() object

    Returns:
        a keras layer - resnet
    """
    x = keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        x = ResidualUnit(filters, strides=strides)(x)
        prev_filters = filters

    # since the last res units have 512 feature maps, x should have length 512 after global pooling
    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Flatten()(x)
    return x


def convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        padding="SAME",
        normalization_type=None,
        normalization_kwargs={},
        downsampling_type='conv',
        activation=layers.LeakyReLU,
        name="convnet",
        *args,
        **kwargs):
    normalization_layer = {
        'batch': layers.BatchNormalization,
        'layer': layers.LayerNormalization,
        'group': tfa.layers.normalizations.GroupNormalization,
        'instance': tfa.layers.normalizations.InstanceNormalization,
        None: None,
    }[normalization_type]

    def conv_block(conv_filter, conv_kernel_size, conv_stride):
        block_parts = [
            layers.Conv2D(
                filters=conv_filter,
                kernel_size=conv_kernel_size,
                strides=(conv_stride if downsampling_type == 'conv' else 1),
                padding=padding,
                activation='linear',
                *args,
                **kwargs),
        ]

        if normalization_layer is not None:
            block_parts += [normalization_layer(**normalization_kwargs)]

        block_parts += [(layers.Activation(activation)
                         if isinstance(activation, str)
                         else activation())]

        if downsampling_type == 'pool' and conv_stride > 1:
            block_parts += [getattr(layers, 'AvgPool2D')(
                pool_size=conv_stride, strides=conv_stride)]

        block = tfk.Sequential(block_parts, name='conv_block')
        return block

    def preprocess(x):
        """Cast to float, normalize, and concatenate images along last axis."""
        x = tree.map_structure(
            lambda image: tf.image.convert_image_dtype(image, tf.float32), x)
        x = tree.flatten(x)
        x = tf.concat(x, axis=-1)
        x = (tf.image.convert_image_dtype(x, tf.float32) - 0.5) * 2.0
        return x

    model = tf.keras.Sequential((
        tfkl.Lambda(preprocess),
        *[
            conv_block(conv_filter, conv_kernel_size, conv_stride)
            for (conv_filter, conv_kernel_size, conv_stride) in
            zip(conv_filters, conv_kernel_sizes, conv_strides)
        ],
        tfkl.Flatten(),

    ), name=name)

    return model
