# -*- coding:utf-8 -*-
import sys
import shutil
import os
import jieba
import requests
import numpy as np
import cPickle as pickle
from django.shortcuts import render
from models import ModelInfo,ModelHistory
from django.http import JsonResponse
from keras.models import load_model,Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Embedding,Input,merge,Reshape,Dense,Dropout,Convolution2D,MaxPooling2D
from keras.regularizers import l2

reload(sys)
sys.setdefaultencoding('utf-8')
#参数设置
batch_size = 32 #Number of samples per gradient update
nb_epoch = 200 #the number of times to iterate over the training data arrays
embedding_dim = 100 #嵌入层输出维度
classify_sequence_length = 100 #分类模型sequence最大长度
match_sequence_length = 12 #匹配模型sequence最大长度
save_list = [u'ios', u'app', u'pk', u'tts']
#对问题分词并处理非中文字符
def cut_unchinese(raw_string):
    result = []
    seg_list = jieba.cut(raw_string.lower())
    for seg in seg_list:
        seg = ''.join(seg.split())
        seg = seg.decode(encoding='utf-8')
        seg_result = ''
        if seg in save_list:
            result.append(seg)
        else:
            if seg != u'' and seg != u"\n" and seg != u"\n\n":
                for char in seg:
                    if (char >= u'\u4e00') and (char <= u'\u9fa5'):
                        seg_result += char
                if len(seg_result) > 0:
                    result.append(seg_result)
    return result