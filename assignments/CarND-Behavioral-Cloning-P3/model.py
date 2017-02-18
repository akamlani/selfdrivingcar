import numpy as np

from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential

import keras.backend as K
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, ELU, Lambda, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras.regularizers import l2

import train as tr
import nn_utils as nu

def attr_layer(model, **kwargs):
    if 'batchnorm' in kwargs and kwargs['batchnorm']:
        model.add(BatchNormalization())
    if 'activation' in kwargs and kwargs['activation']:
        model.add(Activation(kwargs['activation']))
    if 'maxpool' in kwargs and kwargs['maxpool']:
        model.add(MaxPooling2D((2, 2), border_mode='valid'))
    if 'dropout_proba' in kwargs and kwargs['dropout_proba']:
        model.add(Dropout(kwargs['dropout_proba']))
    return model

def create_dense(model, layer_name, num_neurons, **kwargs):
    model.add(Dense(output_dim=num_neurons, activation=None, init='uniform', name=layer_name))
    return attr_layer(model, **kwargs)

def create_conv(model, out_depth, layer_name, **kwargs):


    """
    glorot_normal: Gaussian initialization scaled by fan_in + fan_out (Glorot 2010)
    glorot_uniform
    """

    model.add(Convolution2D(out_depth, *kwargs['kernel'],
                            init ='glorot_uniform', name=layer_name,
                            subsample=kwargs['stride'], border_mode='same') )
    return attr_layer(model, **kwargs)


def create_baselayer(image_dim, crop_dim):
    # camera format(dim_ordering=tf): input_shape=(samples, rows(height=y), cols(width=x), channels)
    # apply normalization, cropping within model: take advantage of gpu
    model  = Sequential()
    # Normalize batch at model for faster computation [normalize between: [-0.5,0.5]]
    # normalize/unnormalize steering angles: already normalized in dataset
    model.add(Lambda(lambda x: ((x/255.0) - 0.5), input_shape=image_dim, output_shape=image_dim))
    # crop region to focus on primary focus
    model.add(Cropping2D(cropping=crop_dim))
    # reize causes memory here, so handle it in the batch generator
    return model

def create_model_tl(learning_rate=1e-4, loss_metric='mse', **kwargs):
    """
    from keras.applications import vgg16
    from keras import backend as K

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    img = vgg16.preprocess_input(img)

    # save bottleneck features and reuse
    bottleneck_features_train = model.predict_generator(generator, 2000)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    bottleneck_features_validation = model.predict_generator(generator, 800)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    train_data = np.load(open('bottleneck_features_train.npy'))
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    """

def create_model_custom(learning_rate=1e-4, loss_metric='mse', **kwargs):
    """
    Create a custom model based on nvidia model
    """
    activation = 'elu'
    image_dim, crop_dim = (160,320,3), ((75,25),(10,10))
    model = create_baselayer(image_dim, crop_dim)

    # Conv1: Kernel: 3,3; Stride=1; Input depth: 3,  Output Depth: 24
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': activation, 'batchnorm': True}
    model = create_conv(model, out_depth=24, layer_name='conv1', **params)
    # Conv2: Kernel: 3,3; Stride=1; Input depth: 24,  Output Depth: 36
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': activation, 'batchnorm': True, 'maxpool': True}
    model = create_conv(model, out_depth=36, layer_name='conv2', **params)
    # Conv3: Kernel: 3,3; Stride=1; Input depth: 36,  Output Depth: 48
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': activation, 'batchnorm': True}
    model = create_conv(model, out_depth=48, layer_name='conv3', **params)
    # Conv4: Kernel: 3,3; Stride=1; Input depth: 48,  Output Depth: 72
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': activation, 'batchnorm': True, 'maxpool': True}
    model = create_conv(model, out_depth=72, layer_name='conv4', **params)
    # Conv5: Kernel: 3,3; Stride=1; Input depth: 48,  Output Depth: 72
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': activation, 'batchnorm': True}
    model = create_conv(model, out_depth=96, layer_name='conv5', **params)

    # Flatten number of ndurons should be 1164 per NVIDIA model
    model.add(Flatten())
    # dropout probability: tensorflow: percent=proba_keep, keras=p1-proba_keep
    params = {'activation': activation, 'batchnorm': True}#, 'dropout_proba': 0.25}
    model = create_dense(model, layer_name='dense_1', num_neurons=100, **params)
    model = create_dense(model, layer_name='dense_2', num_neurons=50,  **params)
    model = create_dense(model, layer_name='dense_3', num_neurons=10,  **params)
    # output layer: number of neurons=1; no softmax layer is used in last FC layer
    model.add(Dense(1, name='dense_output'))
    # compile the model
    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss=loss_metric, metrics=['accuracy'])
    return model


def create_model_nvidia(learning_rate=1e-4, loss_metric='mse', **kwargs):
    """
    nvidia paper:
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/
    """
    # apply normalization, cropping layers
    image_dim, crop_dim = (160,320,3), ((75,25),(10,10))
    model = create_baselayer(image_dim, crop_dim)

    # Conv1: Kernel: 5,5; Stride=2; Input depth: 3,  Output Depth: 24
    params = {'kernel':(5,5), 'stride':(2,2), 'activation': 'elu'}
    model = create_conv(model, out_depth=24, layer_name='conv1', **params)
    # Conv2: Kernel: 5,5; Stride=2; Input depth: 24, Output Depth: 36
    model = create_conv(model, out_depth=36, layer_name='conv2', **params)
    # Conv3: Kernel: 5,5; Stride=2; Input depth: 36, Output Depth: 48
    model = create_conv(model, out_depth=48, layer_name='conv3', **params)
    # Conv4: Kernel: 3,3; Stride=1; Input depth: 48, Output Depth: 64
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': 'elu'}
    model = create_conv(model, out_depth=64, layer_name='conv4', **params)
    # Conv5: Kernel: 3,3; Stride=1; Input depth: 64, Output Depth: 64
    model = create_conv(model, out_depth=64, layer_name='conv5', **params)

    # Flatten number of ndurons should be 1164 per NVIDIA model
    model.add(Flatten())
    # dropout probability: tensorflow: percent=proba_keep, keras=p1-proba_keep
    params = {'activation': 'elu'}
    model = create_dense(model, layer_name='dense_1', num_neurons=100, **params)
    model = create_dense(model, layer_name='dense_2', num_neurons=50,  **params)
    model = create_dense(model, layer_name='dense_3', num_neurons=10,  **params)
    # output layer: number of neurons=1; no softmax layer is used in last FC layer
    model.add(Dense(1, name='dense_ouput'))

    # compile model: commai used defaults
    opt = Adam(lr=learning_rate)
    model.compile(optimizer="adam", loss=loss_metric, metrics=['accuracy'])
    return model


def create_model_commaai(learning_rate=1e-4, loss_metric='mse', **kwargs):
    """
    Comma.ai paper: https://arxiv.org/pdf/1608.01230.pdf
    Comma.ai code:  https://github.com/commaai/research/blob/master/train_steering_model.py
    """
    image_dim, crop_dim = (160,320,3), ((75,25),(10,10))
    model = create_baselayer(image_dim, crop_dim)
    # convolutional layers
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    # Dense Layers
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    # compile model: commai used defaults
    model.compile(optimizer="adam", loss=loss_metric, metrics=['accuracy'])
    return model





if __name__ == '__main__':
    data_udacity     = 'data/udacity/'
    data_training    = 'data/training/'

    # ALT: Augment for a balanced training set
    # load augmented training data (via execution of data_augment.py)
    X_train, y_train, X_val, y_val = nu.load_partitions(data_training)
    train, validation = ((X_train, y_train), (X_val, y_val))

    model_nvidia  = create_model_nvidia(learning_rate=1e-3)
    #model_commaai = create_model_commaai(learning_rate=1e-3)
    #model_custom  = create_model_custom(learning_rate=1e-3)

    # Train the model w/generators
    params = {'num_epochs': 10, 'batch_size': 32}
    tr.train(model_nvidia, train, validation, model_name='model_training_tracks_aug_nvidia', **params)

    #tr.save_architecture(model)
    #model_custom.summary()
