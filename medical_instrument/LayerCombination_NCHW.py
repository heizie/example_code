from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate,\
                                    Conv2DTranspose, ReLU, Add
from tensorflow.math import sigmoid
"""
The shape of the matrix is alway (m x n)
Version NCHW
"""


def CBR(x, output_shape, kernel_size=[1, 1]):
    """
    Convultion WITHOUT strides
    :param output_shape: output shape [batch_num, m, n, channels]
    :param kernel_size: m and of a kernel: [m, n]=[1, 1]
    """
    conv = Conv2D(filters=output_shape[1],
                  kernel_size=kernel_size,
                  strides=[1, 1],
                  padding='SAME',
                  data_format='channels_first')(x)
    bn = BatchNormalization(axis=1)(conv)
    relu = ReLU()(bn)
    return relu


def SBR(x, output_shape, kernel_size=[2, 2], kernel_stride=[2, 2]):
    """
    Convultion with strides
    :param output_shape: output shape [batch_num, m, n, channels]
    :param kernel_size: m and of a kernel: [m, n]=[2, 2]
    :param kernel_stride: The strides along the height and width: [h, w]=[2, 2]
    """
    conv = Conv2D(filters=output_shape[1],
                  kernel_size=kernel_size,
                  strides=kernel_stride,
                  padding='SAME',
                  data_format='channels_first')(x)
    bn = BatchNormalization(axis=1)(conv)
    relu = ReLU()(bn)
    return relu


def DBR(x, output_shape, kernel_size=[2, 2], kernel_stride=[2, 2]):
    """
    Deconvultion with strides
    :param output_shape: output shape [batch_num, m, n, channels]
    :param kernel_size: m and of a kernel: [m, n]=[2, 2]
    :param kernel_stride: The strides along the height and width: [h, w]=[2, 2]
    """
    deconv = Conv2DTranspose(filters=output_shape[1],
                             kernel_size=kernel_size,
                             strides=kernel_stride,
                             padding='SAME',
                             data_format='channels_first')(x)
    bn = BatchNormalization(axis=1)(deconv)
    relu = ReLU()(bn)
    return relu


def CBS(x, output_shape):
    """
    End with Sigmoid
    """
    conv = Conv2D(filters=output_shape[1],
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='SAME',
                  data_format='channels_first')(x)
    bn = BatchNormalization(axis=1)(conv)
    sig = sigmoid(bn)
    return sig


def CB(x, output_shape):
    """
    End with BathNormalization
    """
    conv = Conv2D(filters=output_shape[1],
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='SAME',
                  data_format='channels_first')(x)
    bn = BatchNormalization(name='y')(conv)
    return bn


def down_sample(x):
    """
    For layer mix type: Branch SBR + Branch CBR + CBR in detection subnetwork
    input[batch_num, channels, m, n, ]
    output[batch_num, channels, m, n]
    """
    input_shape = x.get_shape()
    batch_num = input_shape[0]
    h = input_shape[1] / 2
    w = input_shape[2] / 2
    c = input_shape[3]
    out_shape = [batch_num, c, h, w]

    # Branch SBR
    x1 = SBR(x, output_shape=out_shape)
    x2 = SBR(x, output_shape=out_shape)
    # Branch CBR
    x1 = CBR(x1, output_shape=out_shape,
             kernel_size=[3, 3])

    x2 = CBR(x2, output_shape=out_shape,
             kernel_size=[3, 3])
    # Concatenate x1 and x2, output c = c * 2
    concate = concatenate(inputs=[x1, x2], axis=1)
    # CBR as output
    output = CBR(concate, output_shape=out_shape)
    return output


def up_sample(x, high_feature=False):
    """
    For layer mix type: Branch DBR + Branch CBR + CBR in detection subnetwork
    input[batch_num, m, n, c]
    output[batch_num, m, n, channels]
    """
    
    input_shape = x.get_shape()
    batch_num = input_shape[0]
    h = input_shape[1] * 2
    w = input_shape[2] * 2
    c = input_shape[3] / 4
    out_shape = [batch_num, c, h, w]
    # Branch DBR
    x1 = DBR(x, output_shape=out_shape)
    x2 = DBR(x, output_shape=out_shape)
    # Branch CBR
    x1 = CBR(x1, output_shape=out_shape, kernel_size=[3, 3])
    x2 = CBR(x2, output_shape=out_shape, kernel_size=[3, 3])
    # Concatenate
    x = concatenate(inputs=[x1, x2], axis=1)
    # Concatenate with the features from high level
    if type(high_feature) != type(False):
        high_feature = Conv2D(filters=c*2,
                              kernel_size=[1, 1],
                              strides=[1, 1],
                              padding='SAME',
                              data_format='channels_first')(high_feature)
        high_feature = BatchNormalization(axis=1)(high_feature)
        x = Add()([x, high_feature])
        #x = concatenate(inputs=[concate, high_feature], axis=3)
    # Reduce the channels from 2c to c with CBR
    output = CBR(x, output_shape=[batch_num, input_shape[3] / 2, h, w])
    return output


def out_branch(x, c1, c2):

    input_shape = x.get_shape()
    batch_num = input_shape[0]
    h = input_shape[1]
    w = input_shape[2]
    # Branch DBR
    x1 = DBR(x,
             output_shape=[batch_num, c1, h, w],
             kernel_size=[1, 1],
             kernel_stride=[1, 1])
    x2 = DBR(x,
             output_shape=[batch_num, c2, h, w],
             kernel_size=[1, 1],
             kernel_stride=[1, 1])
    # Branch CBR
    y1 = CBR(x1,
             output_shape=[batch_num, c1, h, w],
             kernel_size=[1, 1])
    y2 = CBR(x2,
             output_shape=[batch_num, c2, h, w],
             kernel_size=[1, 1])
    # Branch CBS
    y1 = CBS(y1, output_shape=[batch_num, c1, h, w])
    y2 = CBS(y2, output_shape=[batch_num, c2, h, w])
    return y1, y2


def reg_sub(x):

    input_shape = x.get_shape()
    batch_num = input_shape[0]
    h = input_shape[1]
    w = input_shape[2]
    c = 64

    x = CBR(x, output_shape=[batch_num, c, h, w], kernel_size=[3, 3])
    x = CBR(x, output_shape=[batch_num, c*2, h, w], kernel_size=[3, 3])
    x = CBR(x, output_shape=[batch_num, c*4, h, w], kernel_size=[3, 3])
    x = CBR(x, output_shape=[batch_num, c*4, h, w], kernel_size=[3, 3])
    x = CBR(x, output_shape=[batch_num, c*4, h, w])
    y = CB(x, output_shape=[batch_num, 7, h, w])
    return y

