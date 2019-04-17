#!/usr/bin/python
# -*- coding: utf-8 -*-

from rnn_model import *
# from data.cnews_loader import *
from sklearn import metrics
import numpy as np
import sys
import os
import time
from datetime import timedelta
from tensorflow.contrib import learn
import codecs
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


base_dir = 'E:/stockcomment/'
data_dir = os.path.join(base_dir, 'train_stockcomment_1fenci.txt')
label_dir = os.path.join(base_dir, 'label.txt')
embed_dir = os.path.join(base_dir, 'vectors.txt')
# vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
test_dir = 'E:/stockcomment/twitter_test.txt'
predict_dir = 'E:/stockcomment/predict.txt'

save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def batch_iter(x, y, batch_size=50):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def batch_iter_x(x, batch_size=50):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x[start_id:end_id]

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 50)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def load_data_and_labels(data_dir, label_dir):
    with codecs.open(data_dir, 'r', encoding='utf8') as f:
        data = f.readlines()
    with codecs.open(label_dir, 'r', encoding='utf8') as f:
        labels = f.readlines()
    labels = np.array(labels, dtype='float')
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in data])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    x = np.array(list(vocab_processor.fit_transform(data)))

    # One-hot encoding
    enc = preprocessing.OneHotEncoder()
    y = enc.fit_transform(labels.reshape(len(labels), 1)).toarray()

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test, vocab_processor

def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    # x_train, y_train = process_file_train(word_to_id, cat_to_id, config.seq_length)
    # x_val, y_val = process_file_val(word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0              # 总批次
    best_acc_val = 0.0           # 最佳验证集准确率
    last_improved = 0            # 记录上一次提升批次
    require_improvement = 1000   # 如果超过1000轮未提升，提前结束训练

    if embed_dir:
        # initial matrix with random uniform
        initW = np.random.uniform(-1,1,(len(vocab_processor.vocabulary_), config.embedding_dim))
        # load any vectors from the word2vec
        print("Load glove file {}\n".format(embed_dir))
        with codecs.open(embed_dir, "r", encoding='utf8') as f:
            header = f.readline()
            #vocab_size, layer1_size = map(int, header.split())
            #binary_len = np.dtype('float32').itemsize * layer1_size
            for line in f.readlines():
                word = line.strip().split(' ')[0]
                vector = line.strip().split(' ')[1:]
                idx = vocab_processor.vocabulary_.get(word)
                if idx != 0:
                    initW[idx] = [np.float32(i) for i in vector]

        session.run(model.embedding.assign(initW))

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)   # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def test():
    print("Loading test data...")
    start_time = time.time()
    # x_test, y_test = process_file_test(word_to_id, cat_to_id, config.seq_length)
    with open(test_dir, 'r', encoding='utf8') as f:
        data = f.readlines()

    x = np.array(list(vocab_processor.fit_transform(data)))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    result = []
    batch_train = batch_iter_x(x, config.batch_size)
    for x_batch in batch_train:
        feed_dict = {
            model.input_x: x_batch,
            # model.input_y: y,
            model.keep_prob: 1.0
        }
        y_pred_cls = session.run(model.y_pred_cls, feed_dict=feed_dict)
        result.extend(y_pred_cls)

    label_dict = {
        0: 0, 1: 1
    }

    with codecs.open(predict_dir, 'w', encoding='utf8') as f:
        for label, content in zip(result, data):
            f.write(str(label_dict[label]))
            f.write("\n")

    # print('Testing...')
    # loss_test, acc_test = evaluate(session, x_test, y_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))

    # batch_size = 128
    # data_len = len(x_test)
    # num_batch = int((data_len - 1) / batch_size) + 1
    #
    # y_test_cls = np.argmax(y_test, 1)
    # y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    # for i in range(num_batch):   # 逐批次处理
    #     start_id = i * batch_size
    #     end_id = min((i + 1) * batch_size, data_len)
    #     feed_dict = {
    #         model.input_x: x_test[start_id:end_id],
    #         model.keep_prob: 1.0
    #     }
    #     y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
    #
    # # 评估
    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    #
    # # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    # print(cm)
    #
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_rnn.py [train / test]""")

    print('Configuring RNN model...')
    config = TRNNConfig()
    # if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    #     build_vocab_excel_zifu(vocab_dir, config.vocab_size)
    # categories, cat_to_id = read_category()
    # words, word_to_id = read_vocab(vocab_dir)
    # config.vocab_size = len(words)
    x_train, x_val, y_train, y_val, vocab_processor = load_data_and_labels(data_dir, label_dir)
    config.vocab_size = len(vocab_processor.vocabulary_)
    config.seq_length = x_train.shape[1]
    config.num_classes = y_train.shape[1]
    model = TextRNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()
