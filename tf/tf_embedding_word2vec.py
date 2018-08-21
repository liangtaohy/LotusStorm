from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import threading
import time
import zipfile
from six.moves import urllib
from six.moves import xrange
from tempfile import gettempdir
import numpy as np
import tensorflow as tf
import random
import math


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))


if current_path is None:
    current_path = "./embedding_log"


# 步骤1：下载样本数据
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
    print(url + filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified", filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')

    return local_filename


#filename = maybe_download('text8.zip', 31344016)

filename = "/Users/xlegal/Downloads/text8.zip"

# 数据读入list
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

print(vocabulary)


# 步骤2：构建词典，把不常见的词替换为UNK
vocabulary_size = 50000


def build_dataset(words, n_words):
    """构建数据集"""

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0: # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# 步骤3：为skip-gram模型生成训练批次
# skip window:
# 输入为：([context_word], target), context_word<=2*skip_window
# 窗口滑动：每次滑动一个词的距离
# 输出为：(target, context_words_i), i属于[context_word], 随机采样，采样大小为num_skips
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0  # batch_size必须为num_skips的整数倍
    assert num_skips <= 2 * skip_window  # num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin

    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
        reverse_dictionary[labels[i, 0]])

print(batch)
print(labels)
# 步骤4：构建并训练skip-gram模型.

batch_size = 128  # 批次大小，一个批次128个样本
embedding_size = 128  # embedding向量的维度
skip_window = 1  # 左右两侧的单词数
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # 负样本数量

# 我们选择一个随机验证集来对最近邻居进行采样。 在这里我们限制了
# 验证样本到具有低数字ID的单词，其中
# 施工也是最常见的。 这3个变量仅用于
# 显示模型精度，它们不会影响计算。
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():

    # 输入数据
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # NCE Loss
    with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embedding_size],
                stddev=1.0 / math.sqrt(embedding_size)
            )
        )

    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 计算平均NCE loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size
            )
        )

    tf.summary.scalar('loss', loss)

    # 优化器
    # 随机梯度下降，学习速率为1.0
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 在minibatch和all embeddings之间计算余弦相似度
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True
    )

    # 合并summaries
    merged = tf.summary.merge_all()

    # 初始化变量
    init = tf.global_variables_initializer()

    # 创建saver
    saver = tf.train.Saver()

    # 步骤5：开始训练
    num_steps = 100001

    with tf.Session(graph=graph) as session:
        # 保存summary
        writer = tf.summary.FileWriter(current_path, session.graph)

        session.run(init)

        print("初始化完成")


        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # 定义metadata变量
            run_metadata = tf.RunMetadata()

            #
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata
            )

            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    print(-sim[i, :])
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()

        # Write corresponding labels for the embeddings.
        with open(current_path + '/metadata.tsv', 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        # Save the model for checkpoints.
        saver.save(session, os.path.join(current_path, 'model.ckpt'))
        from tensorflow.contrib.tensorboard.plugins import projector
        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(current_path, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

    writer.close()


    # 步骤6：embbeddings可视化
    def plot_with_labels(low_dim_embs, labels, filename):
      assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
      plt.figure(figsize=(18, 18))  # in inches
      for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

      plt.savefig(filename)


    try:
      # pylint: disable=g-import-not-at-top
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt

      tsne = TSNE(
          perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
      plot_only = 500
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
      labels = [reverse_dictionary[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

    except ImportError as ex:
      print('Please install sklearn, matplotlib, and scipy to show embeddings.')
      print(ex)