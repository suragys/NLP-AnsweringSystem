from __future__ import print_function

import collections
import math
import random
import numpy as np
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE
import preprocess_sample_3 as pre_pro



class SkipGram(object):
    def __init__(self):
        self.data_index = 0
        self.final_embeddings = None


    def build_dataset(self, words):
        count = [['UNK', -1]]
        # count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        # instead fo limiting the count to vocabulary size counting the complete word list
        count.extend(collections.Counter(words).most_common())
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        # Initializing the vocabulary size to number of unique words in the corpous
        self.vocabulary_size = len(count)
        self.data = data
        self.count = count
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        return data, count, dictionary, reverse_dictionary



    # do we have to rewrite this code for the sake of plagarism
    def generate_batch(self, batch_size, num_skips, skip_window):
        # global data_index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels


    # what is happenning here why this?
    def run_skip_gram(self):
        for num_skips, skip_window in [(2, 1), (4, 2)]:
            data_index = 0
            batch, labels = self.generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
            # print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
            # print('    batch:', [self.reverse_dictionary[bi] for bi in batch])
            # print('    labels:', [self.reverse_dictionary[li] for li in labels.reshape(8)])

            batch_size = 128
            embedding_size = 300  # Dimension of the embedding vector.
            skip_window = 1  # How many words to consider left and right.
            num_skips = 2  # How many times to reuse an input to generate a label.
            valid_size = 16  # Random set of words to evaluate similarity on.
            valid_window = 100  # Only pick dev samples in the head of the distribution.
            valid_examples = np.array(random.sample(range(valid_window), valid_size))
            num_sampled = 64  # Number of negative examples to sample.

            graph = tf.Graph()

            with graph.as_default(), tf.device('/cpu:0'):
                # Input data.
                train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, embedding_size], -1.0, 1.0))
                print(embeddings.get_shape())
                softmax_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, embedding_size],
                                                                  stddev=1.0 / math.sqrt(embedding_size)))
                softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

                # Model.
                # Look up embeddings for inputs.
                embed = tf.nn.embedding_lookup(embeddings, train_dataset)
                # Compute the softmax loss.
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed, train_labels,
                                                                 num_sampled, self.vocabulary_size))

                optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)


                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
                valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
                similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

                num_steps = 100001
                # num_steps = 10001

                with tf.Session(graph=graph) as session:
                    session.run(tf.initialize_all_variables())
                    average_loss = 0
                    for step in range(num_steps):
                        batch_data, batch_labels = self.generate_batch(batch_size, num_skips, skip_window)
                        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                        average_loss += l
                        if step % 2000 == 0:
                            if step > 0:
                                average_loss = average_loss / 2000
                            print('Average loss at step %d: %f' % (step, average_loss))
                            average_loss = 0
                        # note that this is expensive (~20% slowdown if computed every 500 steps)
                        if step % 10000 == 0:
                            sim = similarity.eval()
                            for i in range(valid_size):
                                valid_word = self.reverse_dictionary[valid_examples[i]]
                                top_k = 8  # number of nearest neighbors
                                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                                log = 'Nearest to %s:' % valid_word
                                for k in range(top_k):
                                    close_word = self.reverse_dictionary[nearest[k]]
                                    log = '%s %s,' % (log, close_word)
                                print(log)
                    self.final_embeddings = normalized_embeddings.eval()
                    # print(session.run(embeddings[0]))
                    # print(final_embeddings[0])

    def plot(self, num_points):
        labels = [self.reverse_dictionary[i] for i in range(1, num_points + 1)]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        two_d_embeddings = tsne.fit_transform(self.final_embeddings[1:num_points + 1, :])
        assert self.final_embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(20, 20))  # in inches
        for i, label in enumerate(labels):
            x, y = two_d_embeddings[i, :]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()



if __name__ == '__main__':
    o = SkipGram()
    p = pre_pro.Preprocess()
    content, ans = p.read_input("train_dev_data", "train_dev_ans")
    ip_sgm, words = p.generate_ip_for_SGM(content)
    o.build_dataset(words)
    o.run_skip_gram()
    num_points = 200
    o.plot(num_points)