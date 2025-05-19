# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

His code is a very good one for RNN beginners. Feel free to check it out.
"""
import preprocess_sample_3 as pre_pro
import skip_gram
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

o = skip_gram.SkipGram('ip_preprocessing/ip_SGM.zip')
p = pre_pro.Preprocess()
content, ans = p.read_input("train_dev_data", "train_dev_ans")
ip_sgm, words = p.generate_ip_for_SGM(content)

o.build_dataset(words)
o.run_skip_gram()

labels = p.create_labels_for_LSTM(ans)
# labels = modify_labels(labels)
input_embed = p.get_ip_SQA_for_LSTM(o)


def get_batch(batch_size, step):
    s = step * batch_size
    b_x = np.array(input_embed[s:s + (batch_size * n_steps)])
    b_x = b_x.reshape([batch_size, n_steps, n_inputs])
    b_y = np.array(labels[s:s + batch_size])
    # import pdb
    # pdb.set_trace()

    return b_x, b_y


# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 10

n_inputs = 300  # word vector size
n_steps = 300  # time steps / Story + q + A 1-4
n_hidden_units = 512  # neurons in hidden layer
n_classes = 4  # one of the four options(0 - 3 )

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (300, 512)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (512, 4)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (512, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (4, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> ( 128 batch * 300 steps, 300 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # mul
    # X_in = (128 batch * 300 steps, 512 inputs)
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # reshape
    # X_in ==> (128 batch, 300 steps, 512 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    """
     # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)
    """

    # cell
    ##########################################

    # basic LSTM Cell.
    with tf.variable_scope('forward'):
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # lstm cell is divided into two parts (c_state, h_state)
        _init_state_fw = lstm_cell_fw.zero_state(batch_size, dtype=tf.float32)

    # basic LSTM Cell.
    with tf.variable_scope('backward'):
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # lstm cell is divided into two parts (c_state, h_state)
        _init_state_bw = lstm_cell_bw.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    # outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    '''
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell,
        cell_bw=cell,
        dtype=tf.float64,
        sequence_length=X_lengths,
        inputs=X)

        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None, initial_state_fw=None,
                                        initial_state_bw=None, dtype=None, parallel_iterations=None, swap_memory=False,
                                        time_major=False, scope=None) {#bidirectional_dynamic_rnn}
    '''
    try:
        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, X_in, initial_state_fw=_init_state_fw,
                                                  initial_state_bw=_init_state_bw, time_major=False)
    except:
        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, X_in, dtype=tf.float32,
                                                          time_major=False)
        # outputs = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, X_in,
        #                                           initial_state_fw=_init_state_fw,
        #                                           initial_state_bw=_init_state_bw, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))

    # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < 1700:
        batch_xs, batch_ys = get_batch(batch_size, step)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        # print(batch_ys)
        # import pdb
        #
        # pdb.set_trace()
        # print(sess.run([pred], feed_dict={x: batch_xs}))
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1

