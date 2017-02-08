import numpy as np
import pickle
import json
import time
import os
import cv2

from sklearn.utils import shuffle
import matplotlib.image as mpimg

import keras.callbacks as cb
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import model_from_json, Model, Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2

import pydot
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

# Transfer Learning
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_v3 import InceptionV3


def batch_generator(data_path, features, target, batch_size):
    # https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    image_dim = (64,64,3)
    resize_image = lambda img,size: cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    while True:
        X, y = shuffle(features, target)
        n_batches = int(len(X)/batch_size)
        for offset in range(0, len(X), batch_size):
            end = offset + batch_size
            batch_x, batch_y = features[offset:end], target[offset:end]
            # possible to have not even batches
            batch_size = len(batch_y)
            X_im  = np.zeros((batch_size,*image_dim), dtype=np.uint8)
            # resize image per batch
            for idx, sample in enumerate(batch_x):
                X_im[idx,:,:,:]  = resize_image(mpimg.imread(data_path + sample), (64,64) )
            # TBD: perform any augmentation
            yield(X_im, batch_y)


def get_model_callbacks(weights_name='model.h5'):
    # Save the best model as and when created
    # period=number of epochs between checkpoints, monitor=['val_loss', 'val_acc']
    checkpoint = cb.ModelCheckpoint(weights_name,
                                    monitor='val_loss', mode='auto',
                                    save_best_only=True, save_weights_only=False, verbose=1)
    # Terminate condition if model does not improve
    # patience = number of epochs w/no imrovement after which training will be stopped
    # min_delta = minimum change to qualify as improvement
    early_stopping = cb.EarlyStopping(monitor='val_loss',
                                      min_delta=0, patience=10,
                                      mode='auto', verbose=1)
    # Visualizations
    # histogram_freq = frequency in epochs at which to compute activation histogram
    tensorboard = cb.TensorBoard(log_dir='./summaries',
                                 histogram_freq=0, write_graph=True, write_images=False)


    return [checkpoint, early_stopping]


def attr_layer(model, **kwargs):
    if 'batchnorm' in kwargs and kwargs['batchnorm']:
        pass
    if 'activation' in kwargs and kwargs['activation']:
        model.add(Activation(kwargs['activation']))
    if 'maxpool' in kwargs and kwargs['maxpool']:
        model.add(MaxPooling2D((2, 2), border_mode='valid'))
    if 'dropout_proba' in kwargs and kwargs['dropout_proba']:
        model.add(Dropout(kwargs['dropout_proba']))
    return model

def create_dense(model, layer_name, num_neurons, **kwargs):
    model.add(Dense(output_dim=num_neurons, activation=None, name=layer_name))
    model = attr_layer(model, **kwargs)
    return model

def create_model(learning_rate=1e-4, loss_metric='mse', **kwargs):
    model  = Sequential()
    # activation: ReLU vs ELU
    # http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/


    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(64,64,3)  ))
    model.add(Activation('relu'))


    model.add(Flatten())
    # dropout probability: tensorflow: percent=proba_keep, keras=p1-proba_keep
    params = {'activation': 'elu', 'dropout_proba': 0.5}
    model = create_dense(model, layer_name='dense_1', num_neurons=100, **params)
    model = create_dense(model, layer_name='dense_2', num_neurons=50,  **params)
    model = create_dense(model, layer_name='dense_3', num_neurons=10,  **params)
    # output layer: number of neurons=1; no softmax layer is used in last FC layer
    model.add(Dense(1, name='dense_ouput'))
    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss=loss_metric, metrics=['accuracy'])

    return model


def train(train, validation, model, **kwargs):
    data_path = 'data/udacity/'
    train_generator = batch_generator(data_path, *train, kwargs['batch_size'])
    val_generator   = batch_generator(data_path, *validation, kwargs['batch_size'])

    train_start_time = time.time()
    hist = model.fit_generator( generator=train_generator,
                                validation_data=val_generator,
                                samples_per_epoch=len(train[0]),
                                nb_val_samples=len(validation[0]),
                                nb_epoch=kwargs['num_epochs'],
                                callbacks=get_model_callbacks(),
                                verbose=1
    )
    history = hist.history
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print("Training time: {:.2} sec {:.2} min".format(train_time, train_time/60))
    # save model
    save_model(model, history)
    return model

def save_model(model, history, model_name='model'):
    if not os.path.exists('./ckpts'): os.makedirs('./ckpts')
    model_json_name = './ckpts/' + model_name + '.json'
    model_name = './ckpts/'+ model_name +'.h5'
    # Save model weights
    model.save_weights(model_name)
    # Save model config (architecture)
    model_json = model.to_json()
    with open(model_json_name, "w") as f: f.write(model_json)
    print("Model Saved: {}, {}".format(model_name, model_json_name))
    # Save Training history
    with open('training_history.p', 'wb') as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
    # model.save(model_name)
    # model.load_weights(model_name)

def save_architecture(model, model_name='model'):
    # Save Model Visualization
    # conda install GraphViz; pip install pydot3; binstar search -t conda pydot
    model_file = './graphs/model_architecture.png'
    if not os.path.exists('./graphs'): os.makedirs('./graphs')
    plot(model, to_file=model_file, show_shapes=True, show_layer_names=True)



if __name__ == '__main__':
    # nvidia:   http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # comma.ai: https://github.com/commaai/research/blob/master/train_steering_model.py

    # Input Image (RGB) -> YUV Planes
    # Steering Angle Command Output: 1/r
    # Steering Angles: Steer Left=Left Turns(Negative), Drive Straight=Center(0.0), Steer Right=Right Turn(Positive)

    # drive.py: RGB format is sent
    # model.py: cv2.imread: reads in BGR format; mpimg.imread: format <TBD>

    # Input Image Format: Resize/Rescale, e.g. downscale by factor of 2
    # image resize: shape/k
    pass
