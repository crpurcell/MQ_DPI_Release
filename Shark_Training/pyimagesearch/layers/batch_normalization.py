#=============================================================================#
#                                                                             #
# MODIFIED: 05-Apr-2019 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import keras

#-----------------------------------------------------------------------------#
class BatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to
    freeze parameters.

    Original source: https://github.com/broadinstitute/keras-resnet/blob/
                            master/keras_resnet/layers/_batch_normalization.py
    """

    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # Set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # Force test mode if frozen, otherwise use default (training=None)
        if self.freeze:
            kwargs['training'] = False
        return super(BatchNormalization, self).call(*args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config
