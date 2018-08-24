"""
Parameter file.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import random
import tensorflow as tf

tf.flags.DEFINE_boolean('experiment_mode', False, 'Experiment mode is equivalent to testing a pre-trained model.')

tf.flags.DEFINE_string('base_path', '', 'The base path for codes and data.')
# Note: all the following relative addresses should be relative to the base_path.
tf.flags.DEFINE_string('dict_file_name', 'data/robust04-stats.txt', 'Relative address to the collection stats file.')
tf.flags.DEFINE_string('pre_trained_embedding_file_name', 'data/vectors.6B.100d.txt',
                       'Relative address to the pre-trained embedding file. default dim: 100.')

tf.flags.DEFINE_string('log_path', 'tf-log/', 'TensorFlow logging directory.')
tf.flags.DEFINE_string('model_path', 'model/', 'TensorFlow model directory.')
tf.flags.DEFINE_string('result_path', 'results/', 'TensorFlow model directory.')
tf.flags.DEFINE_string('run_name', '', 'A name for the run.')

tf.flags.DEFINE_integer('batch_size', 512, 'Batch size for training. default: 512.')
tf.flags.DEFINE_integer('num_train_steps', 100000, 'Number of steps for training. default: 100000.')
tf.flags.DEFINE_integer('num_valid_steps', 1000, 'Number of steps for training. default: 1000.')
tf.flags.DEFINE_integer('emb_dim', 100, 'Embedding dimensionality for words. default: 100.')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for Adam Optimizer. default: 0.0001.')
tf.flags.DEFINE_float('dropout_parameter', 1.0, 'Dropout parameter. default: 1.0 (no dropout).')
tf.flags.DEFINE_float('regularization_term', 0.0001, 'Dropout parameter. default: 0.0001 (it is not a good value).')

tf.flags.DEFINE_integer('hidden_1', -1, 'Size of the first hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_2', -1, 'Size of the second hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_3', -1, 'Size of the third hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_4', -1, 'Size of the third hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_5', -1, 'Size of the third hidden layer. Should be positive. default: -1.')

tf.flags.DEFINE_integer('validate_every_n_steps', 10000,
                        'Print the average loss value on the validation set at every n steps. default: 10000.')
tf.flags.DEFINE_integer('save_snapshot_every_n_steps', 10000, 'Save the model every n steps. default: 10000.')

tf.flags.DEFINE_integer('max_q_len', 10, 'Maximum query length. default: 10.')
tf.flags.DEFINE_integer('max_doc_len', 1000, 'Maximum document length. default: 1000.')
tf.flags.DEFINE_integer('dict_min_freq', 20, 'minimum collection frequency of terms for dictionary. default: 20')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.run_name == '':
    print('The run_name argument should be given!')
    exit(0)

random.seed(43)

