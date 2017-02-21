import numpy as np

from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential

import keras.backend as K
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, ELU, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras.regularizers import l2

from keras.layers import Input
from keras.layers import AveragePooling2D
from keras.models import Model

from keras.applications.vgg16 import VGG16                  # h,w,ch = (224,224,3)
from keras.applications.vgg19 import VGG19                  # h,w,ch = (224,224,3)
from keras.applications.resnet50 import ResNet50            # h,w,ch = (224,224,3)
from keras.applications.inception_v3 import InceptionV3     # h,w,ch = (299,299,3)
from keras.applications.xception import Xception            # h,w,ch = (299,299,3)

import train as tr
import nn_utils as nu

### configurable implementation utilities
def attr_layer(model, **kwargs):
    """
    create configurable attributes of a block layer
    Some attributes cannot be configured at time of instantiation
    """
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
    """
    create a configurable dense layer via name
    """
    model.add(Dense(output_dim=num_neurons, activation=None, init='uniform', name=layer_name))
    return attr_layer(model, **kwargs)

def create_conv(model, out_depth, layer_name, **kwargs):
    """
    crate a configurable convolutional layer via name
    """
    model.add(Convolution2D(out_depth, *kwargs['kernel'],
                            init ='glorot_uniform', name=layer_name,
                            subsample=kwargs['stride'], border_mode='same') )
    return attr_layer(model, **kwargs)

def create_baselayer(image_dim, crop_dim):
    """
    create a base layer implemntation before model specifics
    implements Normalization and Cropping taking advantage of GPU
    allows for any image size to be fed into model
    """
    # camera format(dim_ordering=tf): input_shape=(samples, rows(height=y), cols(width=x), channels)
    model  = Sequential()
    # Normalize batch at model for faster computation [normalize between: [-0.5,0.5]]
    # normalize/unnormalize steering angles: already normalized in dataset
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=image_dim, output_shape=image_dim))
    # crop region to focus on primary focus
    model.add(Cropping2D(cropping=crop_dim))
    # TBD: reize causes memory here, so handle it in the batch generator
    return model

### Transfer Learning Models: Not currently used
def prediction_layer(layer_in, nb_classes, classification):
    act = 'softmax' if classification else None
    pred = Dense(nb_classes, activation=act, name='predictions')(layer_in)
    return pred

def train_xception(inp, nb_classes, **kwargs):
    x = GlobalAveragePooling2D(inp)
    return prediction_layer(x, nb_classes)

def train_inceptionv3(inp, nb_classes, **kwargs):
    # Good for real-time performance
    #x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(inp)
    #x = Flatten(name='flatten')(x)
    x = GlobalAveragePooling2D()(inp)
    x = Dense(1024, activation='relu')(x)
    return prediction_layer(x, nb_classes)

def train_resnet50(inp, nb_classes, **kwargs):
    x = Flatten()(inp)
    return prediction_layer(x, nb_classes)

def train_vgg(inp, nb_classes, **kwargs):
    base_layer = Flatten()(inp)
    top_fc1 = Dense(4096, activation='relu', name='fc1')(base_layer)
    top_fc1 = Dropout(0.25)(top_fc1)
    top_fc2 = Dense(4096, activation='relu', name='fc2')(base_layer)
    top_fc2 = Dropout(0.25)(top_fc2)
    return prediction_layer(top_fc2, nb_classes, **kwargs)

def create_model_architecture(model_name, layer_name):
    models = {'vgg16': (VGG16, train_vgg), 'vgg19': (VGG16, train_vgg),
              'resnet50': (ResNet50, train_resnet50),
              'xception': (Xception, train_xception),
              'inception_v3': (InceptionV3, train_inceptionv3) }
    #(224,224,3) if tl_model.name in ['resnet50', 'vgg16', 'vgg19']
    #(299,299,3) if tl_model.name in ['xception', 'inception_v3']
    h,w,ch = [(299,299,3) if tl_model.name in ['xception', 'inception_v3'] else (224,244,3)][0]
    input_tensor = Input(shape=(h,w,ch))
    # instantiate transfer learning model and extract particular layers
    params = {'weights': 'imagenet', 'include_top': False, 'input_tensor': input_tensor}
    model_fn, fn = models[model_name]
    tl_model   = model_fn(**params)
    base_model = Model(input=tl_model.input, output=tl_model.get_layer(layer_name).output)
    inp = base_model.output
    # classification of appropriate model
    pred = fn(inp, nb_classes=1)
    # mark which layers are trainable and rebuild model
    for layer in base_model.layers: layer.trainable = False
    # setup input and output layers for new model
    model = Model( input=tl_model.input, output=pred, name="_".join(['custom', model_name]) )

    # for idx, layer in enumerate(tl_model.layers): print(idx, layer.name, layer.output_shape)
    # attrs = dict([(idx, layer.name,  layer.output_shape, layer.output) for layer in enumerate(tl_model.layers)])
    # top_model.load_weights(model_weights_name, by_name=True)
    # model.add(top_model)
    return model


#### Core Auto Models
def create_model_custom(learning_rate=1e-4, loss_metric='mse', **kwargs):
    """
    Create a custom model based on nvidia model
    """
    activation = 'elu'
    # apply normalization, cropping layers
    image_dim, crop_dim = (160,320,3), ((75,25),(10,10))
    model = create_baselayer(image_dim, crop_dim)

    # Conv1: Kernel: 5,5; Stride=2; Input depth: 3,  Output Depth: 24
    params = {'kernel':(5,5), 'stride':(2,2), 'activation': activation, 'batchnorm': True}
    model = create_conv(model, out_depth=24, layer_name='conv1', **params)
    # Conv2: Kernel: 5,5; Stride=2; Input depth: 24, Output Depth: 36
    model = create_conv(model, out_depth=36, layer_name='conv2', **params)
    # Conv3: Kernel: 5,5; Stride=2; Input depth: 36, Output Depth: 48
    model = create_conv(model, out_depth=48, layer_name='conv3', **params)
    # Conv4: Kernel: 3,3; Stride=1; Input depth: 48, Output Depth: 64
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': activation, 'batchnorm': True}
    model = create_conv(model, out_depth=64, layer_name='conv4', **params)
    # Conv5: Kernel: 3,3; Stride=1; Input depth: 64, Output Depth: 64
    model = create_conv(model, out_depth=64, layer_name='conv5', **params)

    # Flatten number of ndurons should be 1164 per NVIDIA model
    model.add(Flatten())
    # dropout probability: tensorflow: percent=proba_keep, keras=p1-proba_keep
    params = {'activation': activation, 'batchnorm': True, 'dropout_proba': 0.2}
    model = create_dense(model, layer_name='dense_1', num_neurons=100, **params)
    model = create_dense(model, layer_name='dense_2', num_neurons=50,  **params)
    model = create_dense(model, layer_name='dense_3', num_neurons=10,  **params)
    # output layer: number of neurons=1; no softmax layer is used in last FC layer
    model.add(Dense(1, name='dense_ouput'))

    # compile model: commai used defaults
    opt = Adam(lr=learning_rate)
    model.compile(optimizer="adam", loss=loss_metric, metrics=['accuracy'])
    return model


def create_model_nvidia(learning_rate=1e-4, loss_metric='mse', **kwargs):
    """
    nvidia paper:
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/
    nvidia model normalization: x/255 - .5; Activation: ReLU; Input Shape: (66,200,3)
    """
    activation = 'relu'
    # apply normalization, cropping layers
    image_dim, crop_dim = (160,320,3), ((75,25),(10,10))
    model = create_baselayer(image_dim, crop_dim)

    # Conv1: Kernel: 5,5; Stride=2; Input depth: 3,  Output Depth: 24
    params = {'kernel':(5,5), 'stride':(2,2), 'activation': activation}
    model = create_conv(model, out_depth=24, layer_name='conv1', **params)
    # Conv2: Kernel: 5,5; Stride=2; Input depth: 24, Output Depth: 36
    model = create_conv(model, out_depth=36, layer_name='conv2', **params)
    # Conv3: Kernel: 5,5; Stride=2; Input depth: 36, Output Depth: 48
    model = create_conv(model, out_depth=48, layer_name='conv3', **params)
    # Conv4: Kernel: 3,3; Stride=1; Input depth: 48, Output Depth: 64
    params = {'kernel':(3,3), 'stride':(1,1), 'activation': activation}
    model = create_conv(model, out_depth=64, layer_name='conv4', **params)
    # Conv5: Kernel: 3,3; Stride=1; Input depth: 64, Output Depth: 64
    model = create_conv(model, out_depth=64, layer_name='conv5', **params)

    # Flatten number of ndurons should be 1164 per NVIDIA model
    model.add(Flatten())
    # dropout probability: tensorflow: percent=proba_keep, keras=p1-proba_keep
    params = {'activation': activation}
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
    Comma.ai model normalization: x/127.5 -1.; Activation: ELU, Input Shape: (66,200,3)
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

    #model_nvidia  = create_model_nvidia(learning_rate=1e-3)
    #model_commaai = create_model_commaai(learning_rate=1e-3)
    model_custom  = create_model_custom(learning_rate=1e-3)

    # Train the model w/generators
    # Small batch sizes, e.g. 32 have trouble
    params = {'num_epochs': 10, 'batch_size': 64}
    tr.train(model_custom, train, validation, model_name='model_custom', **params)

    tr.save_architecture(model_custom)
    #model_custom.summary()
