import tensorflow as tf
import numpy as np
import os

from math import ceil
from sklearn.utils import shuffle

def optimize(logits, y, **kwargs):
    with tf.name_scope('loss'):
        cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        loss_operation = tf.reduce_mean(cross_entropy)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = kwargs['optimizer_fn'](learning_rate = kwargs['learning_rate'])
        training_operation = optimizer.minimize(loss_operation, global_step)
        tf.summary.scalar('loss', loss_operation)

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


def load_checkpoints(**kwargs):
    sess = tf.get_default_session()
    if not os.path.exists('./ckpts'): os.makedirs('./ckpts')
    try:
        saver_rst = tf.train.Saver()
        saver_rst.restore(sess, tf.train.latest_checkpoint('ckpts'))
        print('restored checkpoint: {}'.format(kwargs['model']))
        #ckpt = tf.train.get_checkpoint_state('./ckpts')
        #saver_rst.restore(sess, ckpt.model_checkpoint_path)
        #print("Restored checkpoint from: {}".format(ckpt.model_checkpoint_path))
    except:
        print("Failed to restore checkpoint, initializing variables")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


def train(train, validation, tensors, **kwargs):
    history = []

    saver = tf.train.Saver()
    train_dir = os.path.join("summaries", "train")
    train_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())
    validation_dir = os.path.join("summaries", "validation")
    validation_writer  = tf.summary.FileWriter(validation_dir, graph=tf.get_default_graph())

    with tf.Session() as sess:
        load_checkpoints(**kwargs)
        for epoch_idx in range(kwargs['num_epochs']):
            # train the model over sucessive batches
            tr_acc, tr_loss, tr_summaries = process_batches(train, tensors, train_phase=True, **kwargs)
            train_writer.add_summary(tr_summaries, epoch_idx)
            # evaluate validation set on trained model after each epoch
            val_acc, val_loss, val_summaries = process_batches(validation, tensors, train_phase=False, **kwargs)
            validation_writer.add_summary(val_summaries, epoch_idx)
            # TBD: early termination

            if not (epoch_idx % kwargs['log_step']):
                s = "EPOCH: {}, 'Tr Loss': {:.3f}, Val Loss: {:.3f}, Tr Acc: {:.3f}, Val Acc: {:.3f}"
                print(s.format(epoch_idx, tr_loss, val_loss, tr_acc, val_acc))

            # checkpoint (ckpt) model on every epoch iteration
            save_path = saver.save(sess, kwargs['model'] )


def create_feed_dict(tensors, features, labels, train_phase, **kwargs):
    if train_phase:
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

def process_batches(dataset, tensors, train_phase, **kwargs):
    total_acc, total_loss = (0,0)
    sess = tf.get_default_session()

    features, labels = dataset
    n_observations = len(features)
    batch_generator = next_batch(features, labels, **kwargs)
    for batch_x, batch_y in batch_generator:
        feed_dict_data = create_feed_dict(tensors, batch_x, batch_y, train_phase, **kwargs)
        if train_phase:
            op = [kwargs['training_op'], kwargs['loss_op'], kwargs['accuracy_op'], kwargs['log_op']]
            _, loss, accuracy, summaries = sess.run(op, feed_dict=feed_dict_data)
        else:
            op = [kwargs['loss_op'], kwargs['accuracy_op'], kwargs['log_op']]
            loss, accuracy, summaries = sess.run(op, feed_dict=feed_dict_data)
        total_acc  += (accuracy * len(batch_x))
        total_loss += (loss     * len(batch_x))
    # summaries are based on the last batch
    return total_acc/n_observations, total_loss/n_observations, summaries

def next_batch(features, labels, **kwargs):
    n_observations, batch_size = ( len(features), kwargs['batch_size'] )
    n_batches = int(ceil(n_observations/batch_size))
    features, labels = shuffle(features, labels)

    for offset in range(0, n_observations, batch_size):
        end = offset + batch_size
        batch_x, batch_y = features[offset:end], labels[offset:end]
        yield(batch_x, batch_y)

def process_nonbatched(dataset, tensors, phase_str, **kwargs):
    dir_str = os.path.join("summaries", phase_str)
    train_phase = (phase_str == 'train')
    writer = tf.summary.FileWriter(dir_str)
    with tf.Session() as sess:
        load_checkpoints(**kwargs)
        feed_dict_data = create_feed_dict(tensors, *dataset, train_phase, **kwargs)
        op = [kwargs['loss_op'], kwargs['accuracy_op'], kwargs['log_op']]
        loss, acc, summaries = sess.run(op, feed_dict=feed_dict_data)
        writer.add_summary(summaries)
        print("Loss: {:.3f}, Accuracy: {:.3f}".format(loss, acc))
        #acc  = kwargs['accuracy_op'].eval(feed_dict=feed_dict_data, session=sess)
        #loss = kwargs['loss_op'].eval(feed_dict=feed_dict_data, session=sess)

def log_metrics(dataset, tensors, train_phase, **kwargs):
    features, ohe_labels = dataset
    op = [kwargs['loss_op'], kwargs['accuracy_op']]
    with tf.Session() as sess:
        load_checkpoints(**kwargs)
        feed_dict_data = create_feed_dict(tensors, features, labels, train_phase, **kwargs)
        loss, acc = sess.run(ops, feed_dict=feed_dict_data)
    return loss, acc
