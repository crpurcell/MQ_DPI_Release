#=============================================================================#
#                                                                             #
# MODIFIED: 05-Apr-2019 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import os
import warnings
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import DepthwiseConv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Reshape
from keras.layers import Input
from keras.models import Model
import keras.utils as keras_utils
from keras import backend as K
from keras import engine as E
from pyimagesearch.layers import BatchNormalization as BNFreeze
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')

#-----------------------------------------------------------------------------#
class MobileNet:
    """
    Implementation of MobileNet based on the version in Keras Applications.
    Replaces the standard batch normalisation layer with a version that can
    be frozen. This is necessary for performing transfer learning.
    """
    @staticmethod
    def relu6(x):
        return K.relu(x, max_value=6)

    def preprocess_input(x):
        """
        Preprocesses a batch of numpy images in TF mode ([-1,+1] scaling).
        """    
        return imagenet_utils.preprocess_input(x, mode='tf')

    @staticmethod
    def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1),
                             freeze_bn=False):
        """
        Adds an initial convolution layer (with batch normalization and relu6).
        """        
        chanAx = 1 if K.image_data_format() == 'channels_first' else -1
        filters = int(filters * alpha)
        x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
        x = Conv2D(filters, kernel, padding='valid', use_bias=False,
                   strides=strides, name='conv1')(x)
        #x = BatchNormalization(axis=chanAx, name='conv1_bn')(x)
        x = BNFreeze(freeze=freeze_bn, axis=chanAx, name='conv1_bn')(x) 
        return Activation(MobileNet.relu6, name='conv1_relu')(x)

    @staticmethod
    def depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                             depth_multiplier=1, strides=(1, 1), block_id=1,
                             freeze_bn=False):
        """
        Adds a depthwise convolution block.
        """
        
        chanAx = 1 if K.image_data_format() == 'channels_first' else -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
        x = DepthwiseConv2D((3, 3), padding='valid',
                            depth_multiplier=depth_multiplier,
                            strides=strides, use_bias=False,
                            name='conv_dw_%d' % block_id)(x)
        #x = BatchNormalization(axis=chanAx, name='conv_dw_%d_bn' % block_id)(x)
        x = BNFreeze(freeze=freeze_bn, axis=chanAx,
                     name='conv_dw_%d_bn' % block_id)(x)
        x = Activation(MobileNet.relu6, name='conv_dw_%d_relu' % block_id)(x)
        x = Conv2D(pointwise_conv_filters, (1, 1), padding='same',
                   use_bias=False, strides=(1, 1),
                   name='conv_pw_%d' % block_id)(x)
        #x = BatchNormalization(axis=chanAx, name='conv_pw_%d_bn' % block_id)(x)
        x = BNFreeze(freeze=freeze_bn, axis=chanAx,
                     name='conv_pw_%d_bn' % block_id)(x)
        return Activation(MobileNet.relu6, name='conv_pw_%d_relu' % block_id)(x)

    @staticmethod
    def build(input_shape=None, alpha=1.0, depth_multiplier=1,
              dropout=1e-3, include_top=True, weights='imagenet',
              input_tensor=None, pooling=None, classes=1000, freeze_bn=False):
        """
        Instantiates the MobileNet architecture.
        """
        
        if not (weights in {'imagenet', None} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), `imagenet` '
                             '(pre-training on ImageNet), '
                             'or the path to the weights file to be loaded.')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as ImageNet with '
                             '`include_top=True`, `classes` should be 1000')
        
        # Determine proper input shape and default size.
        if input_shape is None:
            default_size = 224
        else:
            if K.image_data_format() == 'channels_first':
                rows = input_shape[1]
                cols = input_shape[2]
            else:
                rows = input_shape[0]
                cols = input_shape[1]

            if rows == cols and rows in [128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=default_size,
                                          min_size=32,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top,
                                          weights=weights)

        if K.image_data_format() == 'channels_last':
            row_axis, col_axis = (0, 1)
        else:
            row_axis, col_axis = (1, 2)
        rows = input_shape[row_axis]
        cols = input_shape[col_axis]

        if weights == 'imagenet':
            if depth_multiplier != 1:
                raise ValueError('If imagenet weights are being loaded, '
                                 'depth multiplier must be 1')

            if alpha not in [0.25, 0.50, 0.75, 1.0]:
                raise ValueError('If imagenet weights are being loaded, '
                                 'alpha can be one of'
                                 '`0.25`, `0.50`, `0.75` or `1.0` only.')

            if rows != cols or rows not in [128, 160, 192, 224]:
                if rows is None:
                    rows = 224
                    warnings.warn('MobileNet shape is undefined.'
                                  ' Weights for input shape '
                                  '(224, 224) will be loaded.')
                else:
                    raise ValueError('If imagenet weights are being loaded, '
                                     'input must have a static square shape '
                                     '(one of (128, 128), (160, 160), '
                                     '(192, 192), or (224, 224)). Input shape '
                                     'provided = %s' % (input_shape,))
                
        if K.image_data_format() != 'channels_last':
            warnings.warn('The MobileNet family of models is only available '
                          'for the input data format "channels_last" '
                          '(width, height, channels). '
                          'However your settings specify the default '
                          'data format "channels_first" '
                          '(channels, width, height).'
                          ' You should set `image_data_format="channels_last"` '
                          'in your Keras config located at ~/.keras/keras.json.'
                          ' The model being returned right now will expect '
                          'inputs to follow the "channels_last" data format.')
            K.set_image_data_format('channels_last')
            old_data_format = 'channels_first'
        else:
            old_data_format = None
            
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = MobileNet.conv_block(img_input, 32, alpha, strides=(2, 2),
                                 freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 64, alpha, depth_multiplier,
                                           block_id=1, freeze_bn=freeze_bn)

        x = MobileNet.depthwise_conv_block(x, 128, alpha, depth_multiplier,
                                           strides=(2, 2), block_id=2,
                                           freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 128, alpha, depth_multiplier,
                                           block_id=3, freeze_bn=freeze_bn)

        x = MobileNet.depthwise_conv_block(x, 256, alpha, depth_multiplier,
                                           strides=(2, 2), block_id=4,
                                           freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 256, alpha, depth_multiplier,
                                           block_id=5, freeze_bn=freeze_bn)

        x = MobileNet.depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                           strides=(2, 2), block_id=6,
                                           freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                           block_id=7, freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                           block_id=8, freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                           block_id=9, freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                           block_id=10, freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                           block_id=11, freeze_bn=freeze_bn)

        x = MobileNet.depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                                           strides=(2, 2), block_id=12,
                                           freeze_bn=freeze_bn)
        x = MobileNet.depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                                           block_id=13, freeze_bn=freeze_bn)

        if include_top:
            if K.image_data_format() == 'channels_first':
                shape = (int(1024 * alpha), 1, 1)
            else:
                shape = (1, 1, int(1024 * alpha))

            x = GlobalAveragePooling2D()(x)
            x = Reshape(shape, name='reshape_1')(x)
            x = Dropout(dropout, name='dropout')(x)
            x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
            x = Activation('softmax', name='act_softmax')(x)
            x = Reshape((classes,), name='reshape_2')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = E.get_source_inputs(input_tensor)
        else:
            inputs = img_input
            
        # Create model.
        model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

        # Load weights
        if weights == 'imagenet':
            if K.image_data_format() == 'channels_first':
                raise ValueError('Weights for "channels_first" format '
                                 'are not available.')
            if alpha == 1.0:
                alpha_text = '1_0'
            elif alpha == 0.75:
                alpha_text = '7_5'
            elif alpha == 0.50:
                alpha_text = '5_0'
            else:
                alpha_text = '2_5'

            if include_top:
                model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
                weight_path = BASE_WEIGHT_PATH + model_name
                weights_path = keras_utils.get_file(model_name,
                                                    weight_path,
                                                    cache_subdir='models')
            else:
                model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
                weight_path = BASE_WEIGHT_PATH + model_name
                weights_path = keras_utils.get_file(model_name,
                                                    weight_path,
                                                    cache_subdir='models')
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)

        if old_data_format:
            K.set_image_data_format(old_data_format)
        return model
