# -*- coding: utf-8 -*-

from scipy.io import wavfile
from python_speech_features import mfcc
# 1.读取wav格式数据
path = './music/data/1KHz-STERO.wav'
(rate,data) = wavfile.read(path)
print("文件的rate值：{}",format(rate))
print("文件的数据的大小：{}",format(data.shape))
print(data[:10])
print('-'*100)

# 提取特征信息
mfcc_feat = mfcc(signal=data,samplerate=rate,numcep=26,nfft=2048)
print(type(mfcc_feat))
print(mfcc_feat)

#
