# -- encoding:utf-8 --
"""
特征处理、特征工程相关API
"""
import os
from pydub import AudioSegment
from python_speech_features import mfcc
import numpy as np
import pandas as pd
from scipy.io import wavfile
import glob

#设置相关路径
#1. 音乐文件路径
#2. 音乐文件对应类型的文件地址（music_info.csv）
#3. 音乐文件转化为mfcc后，特征矩阵文件存放地址
#4. 音乐类型以及其对应编号文件存放地址
music_audio_regex_dir = './data/music/*.mp3'
music_info_csv_file_path = './data/music_info.csv'
music_feature_file_path = './data/music_feature.csv'
music_label_index_file_path = './data/music_index_label.csv'

def extract(file, file_format):
    '''
    从指定的file文件当中提取对应的mfcc特征信息，该函数提取到的是，
    某一个音乐的mfcc特征对应的新特征信息。
    :param file: 对应的文件夹路径
    :param file_format: 文件格式
    :return:
    '''
    result = []
    # 1.如果文件不是wav格式的，那么将其转化为wav格式（实际上只是根据文件名称，来分析是否要转换文件格式）
    is_tmp_file = False
    if file_format != 'wav':
        try:
            #读取数据
            song = AudioSegment.from_file(file, format=file_format)
            #获得输出的文件路径
            wav_file_path = file.replace(file_format, 'wav')
            #数据输出
            song.export(out_f=wav_file_path, format='wav')
            is_tmp_file = True #临时文件
        except Exception as e:
            result=[]
            print('Error:.'+file+'to wav format file returns exception.msg:=',end='')
            print(e)
    else:
        wav_file_path = file # 如果当前文件就是wav格式，那他的路径就是file路径
    #上面我们没有没有读取任何文件，只是做一个路径的判断

    #2. 当前文件地址已经是wav格式音乐的地址了，所以可以直接对其进行提取mfcc
    try:
        #读取wav格式数据
        (rate, data) = wavfile.read(wav_file_path)

        #提取mfcc特征  umcep数据集特征个数  nfft 傅里叶变化参数
        mfcc_feat = mfcc(data, rate, numcep=13, nfft=2048)
        #每一个音乐文件的长度不同，是不是会导致最终每个音乐文件所产生的数据集维度不同，
        # 实际上就是最原始的音频数据，大小，通道，采样频率等指标都不一样，
        # 所以mfcc_feat得到的结果大小就不一致，一般的格式为(?, numcep)（每一首歌的数据维度）。
        # 这样就导致了每个样本的特征信息长短不一，所以这里要对内容进行处理，
        # 生成新的特征，放弃老的特征，使得每一首歌它对应的特征维度都是(1, 新特征个数)，
        # 也就是说，现在的每一首歌的信息从矩阵，变为了向量。
        #解决方案一：在现有的mfcc指标上，提取出更高层次的信息特征：
        # 即均值和协方差。(老的mfcc特征数为13，新的特征数应该为13+13+12+11+……+1=104
        # 使得新特征中既有原始特征的均值也有原始特征的协方差)
        #解决方案二：使用AudioSegment当中的相关API，对音频文件做一个预处理，
        # 使得所有音频数据的长度一致、各个特征信息也一致，
        # 也可以保证提取出来的mfcc特征数目一致（105,12000,13)把数据集拉直，变成(105,12000*13)

        #解决方案一实施方法：
        #1. 矩阵的转置
        mm = np.transpose(mfcc_feat)
        #2 求每行13个特征属性中各个特征属性的均值，协方差矩阵
        mf = np.mean(mm, axis=1)
        cf = np.cov(mm)
        #3. 求13个特征构成的协方差矩阵和均值的合并结果
        #最终的维度结果应该有104个特征：13+13+12+……+1
        result = mf
        for i in range(mm.shape[0]):
            #获取协方差矩阵上对角线上的内容，添加到result中
            result = np.append(result, np.diag(cf, i))
        #4. 返回结果
        return result

    except Exception as e:
        result = []
        print(e)
    finally:
        #如果是临时文件，删除临时文件
        if is_tmp_file:
            os.remove(wav_file_path)

    return result


def extract_label():
    '''
    提取标签数据，返回对应的dict类型的数据
    :return:
    '''
    #1. 读取标签数据，得到dataframe对象
    df = pd.read_csv(music_info_csv_file_path)
    #2. 将Dataframe转化为Dict对象，以name作为key，以tag作为value
    name_label_list = np.array(df).tolist()
    name_label_dict = dict(map(lambda t:(t[0].lower(), t[1]),name_label_list))
    labels = set(name_label_dict.values())
    label_index_dict = dict(zip(labels, np.arange(len(labels))))
    #3. 返回结果
    return name_label_dict, label_index_dict

def extract_and_export_all_music_feature_and_label():
    '''
    提取所有的music的特征属性数据，并输出为csv文件，但是，这个函数，
    对于处理测试集数据处理并不支持，那么我们在下一个函数当中，
    需要定义出一个支持处理新来数据的函数。
    :return:
    '''
    #1. 读取csv文件格式数据得到音乐名称对应的音乐类型组成的dict类型的数据
    name_label_dict, label_index_dict = extract_label()

    #2. 获取所有音频文件对应的路径
    music_files = glob.glob(music_audio_regex_dir)
    music_files.sort()
    #3. 遍历所有音频文件得到特征数据
    flag = True
    music_features = np.array([])
    for file in music_files:
        print('开始处理文件：{}'.format(file))
        #a. 提取文件名称
        music_name_format = file.split('\\')[-1].split('-')[-1].split('.')
        music_name = '.'.join(music_name_format[0:-1]).strip().lower()
        music_format = music_name_format[-1].strip().lower()

        #b. 判断musicname对应的label是否存在，如果存在的，直接可以获取音频文件的mfcc值了，
        # 如果不存在，要把当前这个音乐文件给过滤掉
        if music_name in name_label_dict:
            #c. 获取该文件对应的标签label
            label_index = label_index_dict[name_label_dict[music_name]]
            #d. 获取音频文件对应的mfcc特征属性, ff是一个一维数组
            ff = extract(file, music_format)
            if len(ff) == 0:
                continue
            #e.将标签添加到特征属性之后
            ff = np.append(ff, label_index)
            #f. 将当前音频的信息追加到数组中
            if flag:
                music_features = ff
                flag = False
            else:
                music_features = np.vstack([music_features, ff])
        else:
            print('无法处理：'+file+';原因：找不到对应的歌曲label标签。')

    #4.特征数据储存
    label_index_list= []
    for label in label_index_dict:
        label_index_list.append([label, label_index_dict[label]])
    pd.DataFrame(label_index_list).to_csv(music_label_index_file_path, header=None, index=False, encoding='utf-8')
    pd.DataFrame(music_features).to_csv(music_feature_file_path, header=None, index=False, encoding='utf-8')

    #5. 直接返回
    return music_features

def extract_music_feature(audio_regex_file_path):
    '''
    提取给定字符串对应的音频数据，他的mfcc格式的特征属性矩阵，也就是测试集数据提取。
    :param audio_regex_file_path: 测试集音频文件地址
    :return:
    '''
    #1. 获取文件夹下的所有音乐文件
    all_music_files = glob.glob(audio_regex_file_path)
    all_music_files.sort()
    #2.最终返回的mfcc矩阵
    flag = True
    music_names = np.array([])
    music_features = np.array([])

    for files in all_music_files:
        print('开始处理文件：{}'.format(files))
        #a. 提取文件名称
        music_name_and_format = files.split('\\')[-1].split('-')
        music_name_and_format2 = music_name_and_format[-1].split('.')
        music_name = '-'.join(music_name_and_format[:-1])+'-'+'.'.join(music_name_and_format2[:-1])
        music_format = music_name_and_format2[-1].strip().lower()

        #b.获取音频文件对应的mfcc特征属性，ff是一个一维的数组，他的长度和numcep参数有关
        ff = extract(files, music_format)
        if len(ff) == 0:
            print('提取'+files + '失败')
            continue

        #c.将当前音频的信息追加到数组中
        if flag:
            music_features = ff
            flag = False
        else:
            music_features = np.vstack([music_features, ff])

        #添加文件名称
        music_names = np.append(music_names, music_name)
    return music_names, music_features

def fetch_index_2_label_dict(file_path=None):
    '''
    获取类别id对应类别名称组成的字典对象
    :param file_path: 给定文件路径
    :return:
    '''
    #1.初始化文件
    if file_path is None:
        file_path = music_label_index_file_path
    #2.读取数据
    df = pd.read_csv(file_path, encoding='utf-8', header=None)
    #3. 顺序交换形成dict对象
    label_index_list = np.array(df)
    index_label_dict = dict(map(lambda  t: (t[1], t[0]),label_index_list))
    #4.返回
    return index_label_dict

if __name__ == '__main__':
    extract_and_export_all_music_feature_and_label()
