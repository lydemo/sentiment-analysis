#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy
import sys
import math

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 200      # 词向量维度
    seq_length = 100        # 序列长度
    num_classes = 3       # 类别数
    num_filters = 512        # 卷积核数目
    vocab_size = 5000      # 词汇表达小
    filter_sizes = [3, 4, 5]         # 卷积核尺寸

    hidden_dim = 256       # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 256         # 每批训练大小
    num_epochs = 200         # 总迭代轮次

    print_per_batch = 10    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/device:CPU:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],initializer=tf.random_uniform_initializer())
            # print (self.config.vocab_size)
            # print (self.embedding.shape)
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # print (self.input_x.shape)
            # print (embedding_inputs.shape)
            embedding_inputs_expanddim = tf.expand_dims(embedding_inputs, -1)
            # print (embedding_inputs_expanddim.shape)

        with tf.name_scope("cnn"):
            pooled_outputs = []
            pooled_outputs1 = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                # CNN layer
                conv = tf.layers.conv1d(
                    embedding_inputs,
                    self.config.num_filters,
                    filter_size,
                    name='conv-'+str(filter_size),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=None,
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(0.1),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(0.1)
                )
                conv = tf.layers.batch_normalization(conv)
                # filter_shape = [filter_size, self.config.embedding_dim, self.config.num_filters]
                # W_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # # print (W_1.shape)
                # b_1 = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                # # print (b_1.shape)
                # conv = tf.nn.conv1d(
                #     embedding_inputs,
                #     W_1,
                #     stride=1,
                #     padding="VALID",
                #     name="conv_1")
                # # print(type(embedding_inputs))
                # print (conv.shape)

                # filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                # W_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # print (W_1.shape)
                # b_1 = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                # print (b_1.shape)
                # conv_1 = tf.nn.conv2d(
                #     embedding_inputs_expanddim,
                #     W_1,
                #     strides=[1,1,self.config.embedding_dim,1],
                #     padding="VALID",
                #     name="conv_1")
                # print (conv_1.shape)
                # h_1 = tf.nn.relu(tf.nn.bias_add(conv_1, b_1), name="relu")
                # #Max-pooling
                # pooled_1 = tf.nn.max_pool(
                #     h_1,
                #     ksize=[1, 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name="pool_1")
                # print (pooled_1.shape)
                # sys.exit()




                #global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp-'+str(filter_size))
                gmp = tf.expand_dims(gmp, -1)
                # print (gmp.shape)
                # filter_shape = [filter_size, self.config.num_filters, 1]
                # W_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # print (W_1.shape)
                # b_1 = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                # print (b_1.shape)
                # conv_1 = tf.nn.conv1d(
                #     gmp,
                #     W_1,
                #     stride=1,
                #     padding="VALID",
                #     name="conv_1")

                # gmp_1 = tf.reduce_max(conv_1, reduction_indices=[1], name='gmp-' + str(filter_size))
                # print(type(embedding_inputs))
                #
                # print (gmp_1.shape)
                # print (type(gmp_1))
                # pooled_1 = tf.layers.max_pooling1d(
                #     conv,
                #     2,
                #     1,
                #     padding='valid',
                #     name=None
                # )
                pooled_outputs.append(gmp)


            # Combine all the pooled features
            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            h_pool = tf.concat(pooled_outputs ,1)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])


        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(h_pool_flat, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            print (self.logits.shape)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # print (self.input_y.shape)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
