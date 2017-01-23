import tensorflow as tf
from tensorflow.contrib.layers import flatten

import numpy as np
import os
from math import ceil
from scipy import ndimage
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import seaborn as sns

pad_image      = lambda x, px:  np.pad(x, [[0, 0], [px//2, px//2], [px//2, px//2], [0, 0]], mode="constant")
resize_images  = lambda x, px:  tf.image.resize_images(x, (px,px))
reshape_image  = lambda x, px_size: tf.reshape(x, (-1, px_size, px_size, 1))
flatten_tensor = lambda x, dim: tf.reshape(x_in, [-1, dim])
flatten_dim    = lambda x: x.shape[1]*x.shape[2]*x.shape[3]
create_strides = lambda stride: [1,stride,stride,1]
create_kernel  = lambda ksize:  [1,ksize,ksize,1]

### Format Conversions
# Image Format Required: N(images) x H x W x C(channels)
img_convert_4d_gs = lambda data: tf.reshape(data, [1, data.shape[0], data.shape[1], 1])
# Convolution Kernel Format Required: Kheight x Kwidth x Nchannels(input) X Nchannels(output)
kernel_convert_4d = lambda data, ksize: tf.reshape(data, [ksize, ksize, 1, 1])



def optimize(logits, y, **kwargs):

    # define measurement of error(measure of loss), and minimize the error
    # loss operation ~ cost: sum over all examples in a given batch
    # cross entropy: get delta comparison between logits and true values

    # average cross entropy across images
    # optimizer: SGD variants (stochastic randomized data + mini-batch)
    # optimize parameters: gradient descent (minimize error by following Negative direction of gradient)
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    with tf.name_scope('loss'):
        loss_operation = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss_operation)

    # learning rate: how far along the gradient we should move the parameters
    # learning rate too small: fail to converge; too large: overshoot the minima
    optimizer      = kwargs['optimizer_fn'](learning_rate = kwargs['learning_rate'])

    # operation to perform backpropagation and minimize training loss
    # optimize via back propagation (multiply many partial deratives together via chain rule)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    training_operation = optimizer.minimize(loss_operation, global_step)

    # calculate avg predicteda ccuracy between logits and ohe labels
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy_operation)

    return {
        'optimizer_op':  optimizer,
        'training_op':   training_operation,
        'loss_op':       loss_operation,
        'accuracy_op':   accuracy_operation,
        'log_op':        tf.summary.merge_all()
    }

def alt_optimize(logits, y, **kwargs):
    prediction = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(y * tf.log(prediction + 1e-12), reduction_indices=1)
    optimizer = kwargs['optimizer_fn'](learning_rate = kwargs['learning_rate'])
    training_operation = optimizer.minimize(cross_entropy)
    loss = tf.reduce_mean(cross_entropy)
    cost = tf.reduce_mean(distance(logits, y))
    gradient = np.diff(cost)


def log_metrics(training, validation, placeholders, **kwargs):
    X_train, y_train = training
    X_validation, y_validation = validation
    x,y = placeholders
    # calculate metrics
    sess = tf.get_default_session()
    training_feed_dict   = {x: X_train, y: y_train, dropout_keep_prob: 0.5}
    validation_feed_dict = {x: X_validation, y: y_validation, dropout_keep_prob: 1.0}
    training_acc   = sess.run(kwargs['accuracy_op'], feed_dict=training_feed_dict)
    validation_acc = sess.run(kwargs['accuracy_op'], feed_dict=validation_feed_dict)
    return training_acc, validation_acc


def load_checkpoints(**kwargs):
    sess = tf.get_default_session()
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    try:
        ckpt = tf.train.get_checkpoint_state('./ckpts')
        saver_rst = tf.train.Saver()
        saver_rst.restore(sess, ckpt.model_checkpoint_path)
        print("Restored checkpoint from: {}".format(ckpt.model_checkpoint_path))
    except:
        print("Failed to restore checkpoint, initializing variables")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

def train(train, validation, tensors, **kwargs):
    history = []
    batches, loss_batch, accuracy_batch = ([],[],[])

    saver = tf.train.Saver()
    train_dir = os.path.join("summaries", "train")
    train_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())
    validation_dir = os.path.join("summaries", "validation")
    validation_writer  = tf.summary.FileWriter(validation_dir, graph=tf.get_default_graph())

    with tf.Session() as sess:
        load_checkpoints(**kwargs)
        for epoch_idx in range(kwargs['num_epochs']):
            # train the model over sucessive batches
            tr_acc, tr_loss, tr_summaries = process_batches(train, tensors, 'train', **kwargs)
            train_writer.add_summary(tr_summaries, epoch_idx)
            # evaluate validation set on trained model per epoch
            # Alternatively we could just evaluate entire set split instead of via batch
            val_acc, val_loss, val_summaries = process_batches(validation, tensors, 'dev', **kwargs)
            validation_writer.add_summary(val_summaries, epoch_idx)
            # TBD: early termination

            if not (epoch_idx % kwargs['log_step']):
                s = "EPOCH: {}, 'Tr Loss': {:.3f}, Val Loss: {:.3f}, Tr Acc: {:.3f}, Val Acc: {:.3f}"
                print(s.format(epoch_idx, tr_loss, val_loss, tr_acc, val_acc))
                #* n_batches + batch_idx

                # Log Batch Data per according step size
                # if not (offset % log_batch_step):
                #     previous_batch = batches[-1] if batches else 0
                #     batches.append(log_batch_step + previous_batch)
                #     loss_batch.append(loss)
                #     accuracy_batch.append(accuracy)

            # checkpoint (ckpt) model on every epoch iteration
            save_path = saver.save(sess, kwargs['model'] + '.ckpt' )
    return batches, loss_batch, accuracy_batch

def create_feed_dict(tensors, features, labels, phase, **kwargs):
    if phase == 'train':
        feed_dict = {
            tensors['tensor_features']: features,
            tensors['tensor_yhat']: labels,
            tensors['tensor_phase']: True,
            tensors['tensor_dropout_proba']: kwargs['dropout_proba']
        }
    else:
        feed_dict = {
            tensors['tensor_features']: features,
            tensors['tensor_yhat']: labels,
            tensors['tensor_phase']: False,
            tensors['tensor_dropout_proba']: 1.0
        }
    return feed_dict

def process_batches(train, tensors, phase='train', **kwargs):
    total_loss, total_acc = (0,0)
    features, labels = train
    n_obs = len(features)
    n_batches = int(ceil(features.shape[0]/kwargs['batch_size']))
    batches = next_batch(*train, kwargs['batch_size'])

    sess = tf.get_default_session()
    for batch_idx, (batch_x, batch_y) in enumerate(batches):
        feed_dict = create_feed_dict(tensors, batch_x, batch_y, phase, **kwargs)
        if phase == 'train':
            op = [kwargs['training_op'], kwargs['loss_op'], kwargs['accuracy_op'], kwargs['log_op']]
            _, loss, accuracy, summaries = sess.run(op, feed_dict=feed_dict)
        else:
            op = [kwargs['loss_op'], kwargs['accuracy_op'], kwargs['log_op']]
            loss, accuracy, summaries = sess.run(op, feed_dict=feed_dict)
        total_acc  += (accuracy * len(batch_x))
        total_loss += (loss     * len(batch_x))
    # summaries are based on the last batch
    return total_acc/n_obs, total_loss/n_obs, summaries

def next_batch(features, labels, batch_size):
    n_batches = int(ceil(len(features)/batch_size))
    features, labels = shuffle(features, labels)
    for batch_idx in range(n_batches):
        batch_start = batch_idx*batch_size
        batch_end   = batch_start + batch_size
        batch_features = features[batch_start:batch_start + batch_size]
        batch_labels   = labels[batch_start:batch_start + batch_size]
        yield(batch_features, batch_labels)
        # for offset in range(0, len(features), kwargs['batch_size']):
        #     end = offset + batch_size
        #     batch_x, batch_y = features[offset:end], labels[offset:end]
        #     yield(batch_x, batch_y)


def model_evaluate(testset, tensors, **kwargs):
    test_dir = os.path.join("summaries", "test")
    test_writer = tf.summary.FileWriter(test_dir)
    with tf.Session() as sess:
        load_checkpoints(**kwargs)
        test_feed_dict = create_feed_dict(tensors, *testset, 'test', **kwargs)
        op = [kwargs['loss_op'], kwargs['accuracy_op'], kwargs['log_op']]
        loss, acc, summaries = sess.run(op, feed_dict=test_feed_dict)
        test_writer.add_summary(summaries)
        print("Test Loss: {:.3f}, Accuracy: {:.3f}".format(loss, acc))
        #acc  = kwargs['accuracy_op'].eval(feed_dict=test_feed_dict, session=sess)
        #loss = kwargs['loss_op'].eval(feed_dict=test_feed_dict, session=sess)
        #test_loss, test_accuracy = batch_evaluate(testset, placeholders, **kwargs)

def plot_metrics(batch_data, **kwargs):
    batches, loss_batch, train_acc_batch, valid_acc_batch = batch_data
    # plot loss and accuracies over time
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,8))
    # Plot 1: Loss Plot
    ax1.set_title('Loss')
    ax1.plot(batches, loss_batch, 'g')
    ax1.set_xlim([batches[0], batches[-1]])
    # Plot 2: Accuracies
    ax2.set_title('Accuracy')
    ax2.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    ax2.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
    ax2.set_ylim([0, 1.0])
    ax2.set_xlim([batches[0], batches[-1]])
    ax2.legend(loc='best')
    # final plot configurations
    plt.tight_layout()
    plt.show()
