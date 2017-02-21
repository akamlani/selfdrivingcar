import numpy as np
import pickle
import os
import time
import cv2

from sklearn.utils import shuffle
import matplotlib.image as mpimg

from keras.models import model_from_json, Model
import keras.callbacks as cb

import pydot
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot



def get_model_callbacks(weights_name='model.h5'):
    dirname = './ckpts/train/'
    if not os.path.exists(dirname): os.makedirs(dirname)

    # Save the best model as and when created
    # period=number of epochs between checkpoints, monitor=['val_loss', 'val_acc']
    filename = "_".join([weights_name, 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'])
    filename = os.path.join(dirname, filename)
    checkpoint = cb.ModelCheckpoint(filepath=filename,
                                    monitor='val_loss', mode='auto',
                                    save_best_only=True, save_weights_only=False, verbose=1)
    # Terminate condition if model does not improve
    # patience = number of epochs w/no imrovement after which training will be stopped
    # min_delta = minimum change to qualify as improvement
    early_stopping = cb.EarlyStopping(monitor='val_loss',
                                      min_delta=0, patience=5,
                                      mode='auto', verbose=1)
    # Visualizations
    # histogram_freq = frequency in epochs at which to compute activation histogram
    tensorboard = cb.TensorBoard(log_dir='./summaries',
                                 histogram_freq=0, write_graph=True, write_images=False)


    return [checkpoint, early_stopping]


def batch_generator(features, target, batch_size, **kwargs):
    # https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    image_dim   = (160,320,3)
    resized_dim = (66,200,3)
    # image dimensions: rows, cols; opencv expects cols, rows
    resize_image = lambda img,size: cv2.resize(img, (size[1], size[0]), interpolation = cv2.INTER_AREA)
    while True:
        X, y = shuffle(features, target)
        n_batches = int(len(X)/batch_size)
        for offset in range(0, len(X), batch_size):
            end = offset + batch_size
            batch_x, batch_y = ( np.array(features[offset:end]), np.array(target[offset:end]) )
            # adjust for possible to have not even batches or batches with correct dimensions
            batch_x, batch_y = zip(*[(fname, target) for fname, target in zip(batch_x, batch_y)
                                    if mpimg.imread(fname).shape == image_dim])
            batch_y = np.array(batch_y)
            X_im = np.zeros((len(batch_x),*image_dim), dtype=np.uint8)
            # read in image according to adjusted bath size
            for idx, sample in enumerate(batch_x):
                sample_img = mpimg.imread(sample)
                if sample_img.shape == image_dim: X_im[idx,:,:,:] = sample_img
                #resized_img = resize_image(sample_img, resized_dim[:2])
                #if sample_img.shape == image_dim: X_im[idx,:,:,:] = resized_img

            yield(X_im, batch_y)

def train(model, train, validation, model_name, **kwargs):
    train_generator = batch_generator(*train, kwargs['batch_size'])
    val_generator   = batch_generator(*validation, kwargs['batch_size'])

    train_start_time = time.time()
    hist = model.fit_generator( generator=train_generator,
                                validation_data=val_generator,
                                samples_per_epoch=len(train[0]),
                                nb_val_samples=len(validation[0]),
                                nb_epoch=kwargs['num_epochs'],
                                callbacks=get_model_callbacks(model_name),
                                verbose=1
    )
    history = hist.history
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print("Training time: {:.2} sec {:.2} min".format(train_time, train_time/60))
    # save model
    save_model(model, history, model_name)
    return model

def save_model(model, history, model_name='model'):
    if not os.path.exists('./ckpts'): os.makedirs('./ckpts')
    model_json_name = './ckpts/' + model_name + '.json'
    model_name = './ckpts/'+ model_name +'.h5'
    # Save model weights
    model.save(model_name)
    # Save model config (architecture)
    model_json = model.to_json()
    with open(model_json_name, "w") as f: f.write(model_json)
    print("Model Saved: {}, {}".format(model_name, model_json_name))
    # Save Training history
    with open('ckpts/training_history.p', 'wb') as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

    # model.save_weights(model_name)
    # model.load_weights(model_name)

def save_architecture(model, model_name='model'):
    # Save Model Visualization
    # conda install GraphViz; pip install pydot3; binstar search -t conda pydot
    model_file = './graphs/model_architecture.png'
    if not os.path.exists('./graphs'): os.makedirs('./graphs')
    plot(model, to_file=model_file, show_shapes=True, show_layer_names=True)
