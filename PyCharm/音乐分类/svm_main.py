# -- encoding:utf-8 --
"""
使用svm算法实现模型构建的相关API
"""
import numpy as np
import pandas as pd
import random

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.externals import joblib

import feature_process

#因为可能涉及到随机数
random.seed(28)
np.random.seed(28)
music_feature_file_path = './music/data/music_feature.csv'
music_label_index_file_path = './music/data/music_index_label.csv'
default_model_out_f = './music/data/music_svm.model'

'''
大概需要的算法：
1.模型交叉验证的API，用于模型最优参数的选择
2.具体的模型训练以及模型保存的API
3.使用保存好的模型来预测数据，并产生结果的API
'''
def cross_validation(music_feature_csv_file_path=None, data_percentage=0.7):
    '''
    交叉验证，用于选择模型的最优参数
    :param music_feature_csv_file_path:训练数据的存储文件路径
    :param data_percentage: 给定使用多少数据用于模型选择
    :return:
    '''
    #1.初始化文件路径
    if not music_feature_csv_file_path:
        music_feature_csv_file_path = music_feature_file_path
    #2.读取数据
    print('开始读取原始数据：{}'.format(music_feature_csv_file_path))
    data = pd.read_csv(music_feature_csv_file_path, sep=',', header=None, encoding='utf-8')
    #3.抽取部分数据用于交叉验证
    sample_fact = 0.7
    if isinstance(data_percentage, float) and 0<data_percentage<1:
        sample_fact = data_percentage
    data = data.sample(frac=sample_fact, random_state=28)
    X = data.T[:-1].T
    Y = np.array(data.T[-1:]).reshape(-1)
    print(np.shape(X))  #(67, 104)
    #4.给定交叉验证的参数
    parameters = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [.00001, .0001, .001, .01, .1, 1, 10],
        'degree': [2, 3, 4, 5, 6],
        'gamma': [.00001, .0001, .001, .01],
        'decision_function_shape': ['ovo', 'ovr']
    }
    #5.构建模型并训练
    print('开始进行模型参数选择……')
    model = GridSearchCV(estimator=SVC(random_state=28), param_grid=parameters, cv=3)
    model.fit(X, Y)
    #6. 输出最优的模型参数
    print('最优参数：{}'.format(model.best_params_))
    print('最优的模型得分：{}'.format(model.best_score_))

def fit_dump_model(music_feature_csv_file_path=None, train_percentage=0.7, model_out_f=None, fold=1):
    '''
    进行fold次模型训练，最终将模型效果最好的那个模型保存到model_out_f文件当中去
    :param music_feature_csv_file_path: 训练数据存放的文件路径
    :param train_percentage: 训练数据比例
    :param model_out_f: 模型保存文件路径
    :param fold: 训练过程中，训练的次数
    :return:
    注意：由于现在样本数据有点少，所以模型的衡量指标采用训练集和测试集加权准确率作为衡量指标
    eg：score = 0.35 train_score + 0.65 test_score
    '''

    #1. 变量初始化
    if not music_feature_csv_file_path:
        music_feature_csv_file_path = music_feature_file_path
    if not model_out_f:
        model_out_f = default_model_out_f

    #2. 进行数据读取
    print('开始读取原始数据：{}'.format(music_feature_csv_file_path))
    data = pd.read_csv(music_feature_csv_file_path, sep=',', header=None, encoding='utf-8')
    #3.开始进行循环处理
    max_train_score = None
    max_test_score = None
    max_score = None
    best_model = None
    flag = True
    for index in range(1, int(fold) + 1):
        #3.1 开始进行数据的抽取、分割
        shuffle_data = shuffle(data).T
        X = shuffle_data[:-1].T
        Y = np.array(shuffle_data[-1:]).reshape(-1)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = train_percentage)

        #3.2模型训练
        svm = SVC(kernel='poly', C=1e-5, decision_function_shape='ovo',
                  random_state = 28, degree=2)
        svm.fit(x_train, y_train)
        #3.3获取准确率
        acc_train = svm.score(x_train, y_train)
        acc_test = svm.score(x_test, y_test)
        acc = 0.35 * acc_train + 0.65 * acc_test

        #3.4临时保存最优的模型
        if flag:
            max_score = acc
            max_test_score = acc_test
            max_train_score = acc_train
            best_model = svm
            flag = False
        elif max_score<acc:
            max_score = acc
            max_test_score = acc_test
            max_train_score = acc_train
            best_model = svm
        #3.5打印一下日志信息
        print('第%d次训练，测试集上的准确率为：%.2f, 训练集上的准确率为：%.2f，修正之后的准确率为：%.2f' % (index, acc_test, acc_train, acc))

        #4. 输出最优模型的相关信息
        print('最优模型效果：测试集上准确率为: %.2f, 训练集上准确率为：%.2f，修正之后的准确率为：%.2f' %  (max_test_score, max_train_score, max_score))
        print('最优模型为：')
        print(best_model)

        #5. 模型储存
        joblib.dump(best_model, model_out_f)



def fetch_predict_label(X, model_file_path=None, label_index_file_path=None):
    '''
    获取预测的标签名称
    :param X: 特征矩阵
    :param model_file_path:模型
    :param label_index_file_path:标签id和name的映射文件
    :return:
    '''

    #1. 初始化相关参数
    if not model_file_path:
        model_file_path = default_model_out_f
    if not label_index_file_path:
        label_index_file_path = music_label_index_file_path

    #2.加载模型
    model = joblib.load(model_file_path)
    #3.加载标签的id和name的映射关系
    tag_index_2_name_dict = feature_process.fetch_index_2_label_dict(label_index_file_path)
    #4.做预测数据
    label_index = model.predict(X)

    #5.获取最终标签值
    result = np.array([])
    for index in label_index:
        result = np.append(result, tag_index_2_name_dict[index])

    #6. 返回标签值
    return result

if __name__ == '__main__':
    flag = 1
    if flag == 1:
        '''
        模型做一个交叉验证
        '''
        cross_validation(data_percentage=0.9)
    elif flag ==2:
        '''
        模型训练
        '''
        fit_dump_model(train_percentage=0.9, fold=600)
    elif flag == 3:
        '''
        直接在控制台输出模型预测结果
        '''
        _, X = feature_process.extract_music_feature('./data/test/*.mp3')
        print('X维度：{}'.format(X.shape))
        label_names = fetch_predict_label(X)
        print(label_names)

