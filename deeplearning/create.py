# -*- coding:utf-8 -*-
import sys
import shutil
import os
import jieba
import requests
import numpy as np
import cPickle as pickle
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
app_id = 5
nb_class = 5

#classiy_info = ModelInfo.objects.create(online_url='',offline_url='',type=0,is_online=0,app_id=app_id)
model_dir = os.path.dirname(os.path.abspath(__file__)) + '/model/train/app' + str(app_id)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
classify_url = model_dir + '/classify' + '0' + '.h5'

# build model
with open('/home/wxr/blazer/BotHelperOffline/deeplearning/model/word2vec/embedding.pkl', 'rb') as vocab:
    embedding_matrix = pickle.load(vocab)
nb_words = len(embedding_matrix)
embedding_layer1 = Embedding(input_dim=nb_words,
                             output_dim=embedding_dim,
                             weights=[embedding_matrix],
                             input_length=classify_sequence_length,
                             trainable=False)
embedding_layer2 = Embedding(input_dim=nb_words,
                             output_dim=embedding_dim,
                             weights=[embedding_matrix],
                             input_length=classify_sequence_length,
                             trainable=True)
total_input = Input(shape=(classify_sequence_length,))
model1 = embedding_layer1(total_input)
model1 = Reshape((classify_sequence_length, 100, 1))(model1)

model2 = embedding_layer2(total_input)
model2 = Reshape((classify_sequence_length, 100, 1))(model2)
# merge link tensor ; Merge link model layer
graph_in = merge([model1, model2], mode='concat')

conv_1 = Convolution2D(nb_filter=128, nb_row=1, nb_col=100, border_mode='valid',activation='relu')(graph_in)
conv_2 = Convolution2D(nb_filter=128, nb_row=2, nb_col=100, border_mode='valid',activation='relu')(graph_in)
conv_3 = Convolution2D(nb_filter=128, nb_row=3, nb_col=100, border_mode='valid',activation='relu')(graph_in)

conv_1 = MaxPooling2D(pool_size=(int(conv_1.get_shape()[1]), 1))(conv_1)
conv_2 = MaxPooling2D(pool_size=(int(conv_2.get_shape()[1]), 1))(conv_2)
conv_3 = MaxPooling2D(pool_size=(int(conv_3.get_shape()[1]), 1))(conv_3)
conva = merge([conv_1, conv_2, conv_3], mode='concat', concat_axis=-1)
# model-2
out = Reshape((3 * 128,))(conva)
out = Dense(60, activation='relu', W_regularizer=l2(0.03))(out)
out = Dropout(0.3)(out)
out = Dense(5, activation='softmax')(
    out)
total = Model(input=total_input, output=out)

total.save(classify_url)