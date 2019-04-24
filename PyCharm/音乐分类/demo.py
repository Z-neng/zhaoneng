# -*- coding: utf-8 -*-

from pydub import AudioSegment
import numpy as np
import array

path = './music/data/1KHz-stero.wav'
# path = './music/data/1KHz-STERO.wav'
song = AudioSegment.from_file(file=path,format='wav')
# path = './music/data/我们的纪念.wav'
# song = AudioSegment.from_file(file=path,format='mp3')

song[:3000].export('./music/data/01.wav',format='wav')
song[-3000:].export('./music/data/02.wav',format='wav')
(song+10).export('./music/data/03.wav',format='wav')
(song-10).export('./music/data/04.wav',format='wav')

print(song.channels)
print(song.frame_rate)
print(song.sample_width)

song = song.set_channels(2)
song = song.set_frame_rate(66200)
song = song.set_sample_width(2)
print(song.set_frame_rate)

samples = np.array(song.get_array_of_samples()).reshape(-1)
length = samples.shape[0]
print(length)

if length < 26460000:
    samples = np.pad(samples,(26460000-length,0),mode='constant',constant_values=(0,0))
else:
    samples = samples[:26460000]

print('原始数据大小：{}， 扩展后数据大小：{}'.format(length, samples.shape[0]))

samples_array = array.array(song.array_type,samples)
new_song = song._spawn(samples_array)
new_song.export('./music/data/06.wav', format='wav')
