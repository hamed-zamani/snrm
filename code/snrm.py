"""
The SNRM model proposed in:
Hamed Zamani, Mostafa Dehghani, W. Bruce Croft, Erik Learned-Miller, Jaap Kamps.
"From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing", In CIKM '18.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import numpy as np
import tensorflow as tf

import util


class SNRM(object):
    """
        The implementation of the SNRM model proposed by Zamani et al. (CIKM '18). The model learns a sparse
        representation for query and documents in order to take advantage of inverted indexing at the inference time for
        efficient retrieval. This is the first learning to rank model that does 'ranking' instead of 're-ranking'. For
        more information, please refer to the following paper:

        Hamed Zamani, Mostafa Dehghani, W. Bruce Croft, Erik Learned-Miller, Jaap Kamps.
        "From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing", In CIKM '18.

    """
    def __init__(self, dictionary, pre_trained_embedding_file_name, batch_size, max_q_len, max_doc_len, emb_dim,
                 layer_size, dropout_parameter, regularization_term, learning_rate):
        """
            The SNRM constructor.
            Args:
            dictionary (obj): an instance of the class Dictionary containing terms and term IDs.
            pre_trained_embedding_file_name (str): the path to the pre-trained word embeddings for initialization.
                 This is optional. If a term in the dictionary does not appear in the pre-trained vector file, its
                 embedding will be initialized by a random vector. If this argument is 'None', the embedding matrix will
                 be initialized randomly with a uniform distribution.
            batch_size (int): the batch size for training and validation.
            max_q_len (int): maximum length of a query.
            max_doc_len (int): maximum length of a document.
            emb_dim (int): embedding dimensionality.
            layer_size (list): a list of int containing the size of each layer.
            dropout_parameter (float): the keep probability of dropout. 1 means no dropout.
            regularization_term (float): the weight of the l1 regularization for sparsity.
            learning_rate (float): the learning rate for the adam optimizer.
        """
        self.dictionary = dictionary
        self.pre_trained_embedding_file_name = pre_trained_embedding_file_name
        self.batch_size = batch_size
        self.max_q_len = max_q_len
        self.max_doc_len = max_doc_len
        self.emb_dim = emb_dim
        self.layer_size = layer_size
        self.dropout_parameter = dropout_parameter
        self.regularization_term = regularization_term
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default():
            # The placeholders for input data for each batch.
            self.query_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len])
            self.doc1_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_doc_len])
            self.doc2_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_doc_len])
            self.labels_pl = tf.placeholder(tf.float32, shape=[self.batch_size, 2])

            self.dropout_keep_prob = tf.constant(self.dropout_parameter)

            # For inverted index construction
            self.doc_pl = tf.placeholder(tf.int32, shape=[None, self.max_doc_len])
            self.test_query_pl = tf.placeholder(tf.int32, shape=[None, self.max_q_len])

            # Look up embeddings for inputs. The last element is for padding.
            embeddings = tf.concat([
                self.get_embedding_params(self.dictionary, self.emb_dim, self.pre_trained_embedding_file_name),
                tf.constant(np.zeros([1, self.emb_dim]), dtype=tf.float32)], 0)

            # getting the embedding vectors for the query and the documents.
            emb_layer_q = self.get_embedding_layer_output(embeddings, self.emb_dim, 'emb_layer_query',
                                                                self.query_pl, self.max_q_len)
            emb_layer_d1 = self.get_embedding_layer_output(embeddings, self.emb_dim, 'emb_layer_doc1',
                                                                self.doc1_pl, self.max_doc_len)
            emb_layer_d2 = self.get_embedding_layer_output(embeddings, self.emb_dim, 'emb_layer_doc2',
                                                                self.doc2_pl, self.max_doc_len)

            self.weights, self.weights_name, self.biases, self.biases_name = self.get_network_params(self.layer_size)

            self.q_repr = self.network(emb_layer_q, self.weights, self.weights_name, self.biases, self.biases_name)
            self.d1_repr = self.network(emb_layer_d1, self.weights, self.weights_name, self.biases, self.biases_name)
            self.d2_repr = self.network(emb_layer_d2, self.weights, self.weights_name, self.biases, self.biases_name)

            logits_d1 = tf.reduce_sum(tf.multiply(self.q_repr, self.d1_repr), axis=1, keep_dims=True)
            logits_d2 = tf.reduce_sum(tf.multiply(self.q_repr, self.d2_repr), axis=1, keep_dims=True)
            logits = tf.concat([logits_d1, logits_d2], axis=1)

            # For inverted index construction:
            embedding_layer_doc = self.get_embedding_layer_output(
                embeddings, self.emb_dim, 'emb_layer_doc', self.doc_pl, self.max_doc_len)
            self.doc_representation = self.network(
                embedding_layer_doc, self.weights, self.weights_name, self.biases, self.biases_name)

            # For retrieval:
            embedding_layer_test_query = self.get_embedding_layer_output(
                embeddings, self.emb_dim, 'emb_layer_test_query', self.test_query_pl, self.max_q_len)
            self.query_representation = self.network(
                embedding_layer_test_query, self.weights, self.weights_name, self.biases, self.biases_name)

            # the hinge loss function for training
            self.loss = tf.reduce_mean(
                tf.losses.hinge_loss(logits=logits, labels=self.labels_pl, scope='hinge_loss'))

            # the l1 regularization for sparsity. Since we use ReLU as the activation function, all the outputs of the
            # network are non-negative and thus we do not need to get the absolute value for computing the l1 loss.
            self.l1_regularization = tf.reduce_mean(
                tf.reduce_sum(tf.concat([self.q_repr, self.d1_repr, self.d2_repr], axis=1), axis=1),
                name='l1_regularization')
            # the cost function including the hinge loss and the l1 regularization.
            self.cost = self.loss + (tf.constant(self.regularization_term, dtype=tf.float32) * self.l1_regularization)

            # computing the l0 losses for visualization purposes.
            l0_regularization_docs = tf.cast(tf.count_nonzero(tf.concat([self.d1_repr, self.d2_repr], axis=1)), tf.float32) \
                                     / tf.constant(2 * self.batch_size, dtype=tf.float32)

            l0_regularization_query = tf.cast(tf.count_nonzero(self.q_repr), tf.float32) \
                                      / tf.constant(self.batch_size, dtype=tf.float32)

            # the Adam optimizer for training.
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            # Some plots for visualization
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('l1', self.l1_regularization)
            tf.summary.scalar('l0-docs', l0_regularization_docs)
            tf.summary.scalar('l0-query', l0_regularization_query)

            self.summary_op = tf.summary.merge_all()

            # Add variable initializer.
            self.init = tf.global_variables_initializer()

            # For storing a trained model,
            self.saver = tf.train.Saver()

    def get_network_params(self, layer_sizes):
        """
            Returning the parameters of the network.
            Args:
                layer_sizes (list): a list containing the output size of each layer.

            Returns:
                weights (dict): a mapping from layer name to TensorFlow Variable corresponding to the layer weights.
                weights_name (list): a list of str containing layer names for weight parameters.
                biases (dict): a mapping from layer name to TensorFlow Variable corresponding to the layer biases.
                biases_name (list): a list of str containing layer names for bias parameters.
        """
        weights = {}
        weights_name = ['w' + str(i) for i in xrange(1, len(layer_sizes) - 1)] + ['w_out']

        biases = {}
        biases_name = ['b' + str(i) for i in xrange(1, len(layer_sizes) - 1)] + ['b_out']

        for i in xrange(len(layer_sizes) - 1):
            with tf.name_scope(weights_name[i]):
                weights[weights_name[i]] = \
                    tf.Variable(tf.random_normal([1, 5 if i==0 else 1, layer_sizes[i], layer_sizes[i + 1]],
                                                 name=weights_name[i]))

        return weights, weights_name, biases, biases_name

    def get_embedding_params(self, dictionary, dim, pre_trained_embedding_file_name=None):
        """
            Returning the parameters of the network.
            Args:
                dictionary (obj): an instance of the class Dictionary containing terms and term IDs.
                dim (int): embedding dimensionality.
                pre_trained_embedding_file_name (str): the path to the pre-trained word embeddings for initialization.
                 This is optional. If a term in the dictionary does not appear in the pre-trained vector file, its
                 embedding will be initialized by a random vector. If this argument is 'None', the embedding matrix will
                 be initialized randomly with a uniform distribution.

            Returns:
                embedding_matrix (obj): a 2D TensorFlow Varibale containing the embedding vector for each term ID. For
                 unknown terms, the term_id is zero.
         """
        if pre_trained_embedding_file_name is None:
            return tf.Variable(tf.random_uniform([dictionary.size(), dim], -1.0, 1.0))
        else:
            term_to_id, id_to_term, we_matrix = util.load_word_embeddings(pre_trained_embedding_file_name, dim)
            init_matrix = np.random.random((dictionary.size(), dim))
            for i in xrange(dictionary.size()):
                if dictionary.id_to_term[i] in term_to_id:
                    tid = term_to_id[dictionary.id_to_term[i]]
                    init_matrix[i] = we_matrix[tid]
            return tf.get_variable('embeddings', shape=[dictionary.size(), dim],
                                   trainable=True,
                                   initializer=tf.constant_initializer(init_matrix))

    def get_embedding_layer_output(self, embeddings, dim, layer_name, input, n_terms):
        """
            Getting the output of embedding layer for a batch.
            Args:
                embeddings (obj): a TensorFlow Variable (or Tensor) containing the word embedding vectors.
                dim (int): Embedding dimensionality.
                layer_name (str): a scope name for the embedding layer.
                input (obj): a 2D Tensor (or Placeholder) containing the term ids with the size of batch_size * n_terms.
                n_terms (int): number of terms per instance (text).

            Returns: a 2D Tensor containing the output of the embedding layer for a batch for text.
        """
        with tf.name_scope('embedding_layer'):
            with tf.name_scope(layer_name):
                emb = tf.nn.embedding_lookup(embeddings, tf.reshape(input, [-1]))
                emb = tf.reshape(emb, [-1, 1, n_terms, dim])
        return emb

    def network(self, input_layer, weights, weights_name, biases, biases_name):
        """
            Neural network architecture: a convolutional network with ReLU activations for hidden layers and dropout for
            regularization.

            Args:
                input_layer (obj): a Tensor representing the output of embedding layer which is the input of the neural
                 ranking models.
                weights (dict): a mapping from layer name to TensorFlow Variable corresponding to the layer weights.
                weights_name (list): a list of str containing layer names for weight parameters.
                biases (dict): a mapping from layer name to TensorFlow Variable corresponding to layer biases.
                biases_name (list): a list of str containing layer names for bias parameters.

            Returns: a Tensor containing the logits for the inputs.
        """

        layers = [input_layer]
        for i in xrange(len(weights)):
            with tf.name_scope('layer_' + str(i + 1)):
                # we did not use the biases.
                layers.append(tf.nn.conv2d(layers[i],
                                           weights[weights_name[i]],
                                           strides=[1, 1, 1, 1],
                                           padding='SAME'))

                layers[i + 1] = tf.nn.dropout(
                    tf.nn.relu(layers[i + 1]),
                    self.dropout_keep_prob)

        return tf.reduce_mean(layers[len(layers) - 1], [1, 2])


