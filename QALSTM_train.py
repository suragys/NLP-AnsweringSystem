"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
"""
import preprocess_sample_3 as pre_pro
import skip_gram
import tensorflow as tf
import numpy as np

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

o = skip_gram.SkipGram()
p = pre_pro.Preprocess()
content, ans = p.read_input("train_dev_data", "train_dev_ans")
ip_sgm, words = p.generate_ip_for_SGM(content)

o.build_dataset(words)
o.run_skip_gram()

labels = p.create_labels_for_LSTM(ans)
input_embed = p.get_ip_SQA_for_LSTM(o)


def get_batch(batch_size, step):
    s = step * batch_size
    b_x = np.array(input_embed[s:s + (batch_size * n_steps)])
    b_x = b_x.reshape([batch_size,n_steps,n_inputs])
    b_y = np.array(labels[s:s + batch_size])

    return b_x, b_y


# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 30

n_inputs = 300  # word vector size
n_steps = 300  # time steps / Story + q + A 1-4
n_hidden_units = 512  # neurons in hidden layer
n_classes = 4  # one of the four options(0 - 3 )
total_steps = 1800 # total number of question

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

    # X ==> ((30 batch * 300 steps), 300 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # mul
    # X_in = ((30 batch * 300 steps), 512 inputs)
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # reshape
    # X_in ==> (128 batch, 300 steps, 512 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell

    # basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))

    # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

# prediction for the input batch
pred = RNN(x, weights, biases)
# calculate the entropy loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# optimize the model using Adam Optimizer and optimize the weights
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# Calculate the accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < total_steps:
        batch_xs, batch_ys = get_batch(batch_size, step)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        # Calculate accuracy once for every 10 bactch
        # if step % 10 == 0:
        print(sess.run([accuracy,cost], feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1


# Plot the skip gram plot
num_points = 200
o.plot(num_points)
