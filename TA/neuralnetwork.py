from keras import backend as K
from keras.engine import Layer
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, TimeDistributed

import tensorflow as tf

class NeuralNetwork:
    def vgg(self, grayscale=False):
        if grayscale:
            input_layer = Input(shape=(None, None, 1))
        else:
            input_layer = Input(shape=(None, None, 3))

        nn = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_layer)
        nn = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(nn)
        nn = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(nn)

        nn = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(nn)
        nn = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(nn)
        nn = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(nn)

        nn = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(nn)
        nn = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(nn)
        nn = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(nn)
        nn = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(nn)

        nn = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(nn)
        nn = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(nn)
        nn = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(nn)

        return nn

    def alexnet(self, grayscale=False):
        if grayscale:
            input_layer = Input(shape=(None, None, 1))
        else:
            input_layer = Input(shape=(None, None, 3))

        # layer 1
        nn = Conv2D(96, (11, 11), strides=(4, 4), name='layer1_conv')(input_layer)

        # layer 2
        nn = MaxPooling2D((2, 2), strides=(2, 2), name='layer2_pooling')(nn)
        nn = Conv2D(256, (5, 5), name='layer2_conv1')(nn)
        nn = Conv2D(256, (5, 5), name='layer2_conv2')(nn)
        nn = Conv2D(256, (5, 5), name='layer2_conv3')(nn)

        return nn

    def rpn(self, base_layer, num_anchors):
        nn = Conv2D(512, (3, 3), activation='relu', padding='same', name='rpn_conv1', kernel_initializer='normal')(base_layer)

        classification = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(nn)
        regression = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(nn)

        return [classification, regression]

    def classifier(self, base_layer, input_rois, num_rois, nb_classes = 4):
        out_roi_pool = RoiPoolingConv(7, num_rois)([base_layer, input_rois])

        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)

        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(2))(out)
        out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(2))(out)

        return [out_class, out_regr]

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]   

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)
                

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))