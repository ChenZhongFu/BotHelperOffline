# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jieba
import requests
import numpy as np
import time,datetime
from threading import Thread
import cPickle as pickle
import paramiko
from django.shortcuts import render
from models import ModelInfo,ModelHistory
from django.http import JsonResponse
from keras.models import load_model,Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers import Embedding,Input,Reshape,Dropout,MaxPooling2D,concatenate
from keras import regularizers
import h5py

BASE_DIR = os.path.dirname(__file__)
BASE_DIR = BASE_DIR.replace("BotHelperOffline/deeplearning","")
#参数设置
batch_size = 32 #Number of samples per gradient update
nb_epoch = 20 #the number of times to iterate over the training data arrays
embedding_dim = 100 #嵌入层输出维度
classify_sequence_length = 100 #分类模型sequence最大长度
match_sequence_length = 12 #匹配模型sequence最大长度
save_list = [u'ios', u'app', u'pk', u'tts']
online_path=BASE_DIR
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


#初始化模型
def init(request):
    # 接收参数
    app_id = request.GET.get('appId', '-1')
    nb_class = request.GET.get('classNum','-1')

    if nb_class == '-1' or app_id == '-1':
        return JsonResponse({'retCode': '1000', 'retDesc': '参数错误'})
    old_model=ModelInfo.objects.filter(app_id=app_id,is_online=0)

    if  old_model:
        return JsonResponse({'retCode': '1001', 'retDesc': '已存在模型'})

    classify_info = ModelInfo.objects.create(online_url='',offline_url='',is_online=0,is_training=0,app_id=str(app_id),is_replace=2)
    model_dir = os.path.dirname(os.path.abspath(__file__)) + '/model/train/app' + str(app_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    classify_url = model_dir + '/classify' + str(classify_info.app_id) + '.h5'

    # build model
    with open(BASE_DIR+'embedding.pkl', 'rb') as vocab:
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
    #graph_in = merge([model1, model2], mode='concat')
    graph_in = concatenate(inputs=[model1,model2], axis=-1)
    #conv_1 = Convolution2D(nb_filter=128, nb_row=1, nb_col=100, border_mode='valid',activation='relu')(graph_in)
    #conv_2 = Convolution2D(nb_filter=128, nb_row=2, nb_col=100, border_mode='valid',activation='relu')(graph_in)
    #conv_3 = Convolution2D(nb_filter=128, nb_row=3, nb_col=100, border_mode='valid',activation='relu')(graph_in)

    conv_1 = Conv2D(filters=128, kernel_size=(1,100), strides=(1, 1), padding='valid', activation='relu')(graph_in)
    conv_2 = Conv2D(filters=128, kernel_size=(2,100), strides=(1, 1), padding='valid', activation='relu')(graph_in)
    conv_3 = Conv2D(filters=128, kernel_size=(3,100), strides=(1, 1), padding='valid', activation='relu')(graph_in)

    conv_1 = MaxPooling2D(pool_size=(int(conv_1.get_shape()[1]), 1))(conv_1)
    conv_2 = MaxPooling2D(pool_size=(int(conv_2.get_shape()[1]), 1))(conv_2)
    conv_3 = MaxPooling2D(pool_size=(int(conv_3.get_shape()[1]), 1))(conv_3)
    #conva = merge([conv_1, conv_2, conv_3], mode='concat', concat_axis=-1)
    conva = concatenate(inputs=[conv_1,conv_2,conv_3], axis=-1)

    # model-2
    out = Reshape((3 * 128,))(conva)
    #out = Dense(60, activation='relu', W_regularizer=l2(0.03))(out)
    out = Dense(units=60, activation='relu', kernel_regularizer=regularizers.l2(0.01))(out)
    out = Dropout(0.3)(out)
    #out = Dense(5, activation='softmax')(out)
    out = Dense(units=int(nb_class), activation='softmax')(out)
    total = Model(inputs=total_input, outputs=out)
    sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    total.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    total.save(classify_url)
    del total
    classify_info.offline_url = classify_url
    classify_info.save()
    ModelInfo.objects.create(online_url='',offline_url='',is_online=1,is_training=0,app_id=app_id,is_replace=0)
    return JsonResponse({'retCode': '0', 'retDesc': '新建模型成功','classify':classify_url})
#判断是否可以训练
def isable_train(request):
    app_id = request.GET.get('appId', '-1')
    if app_id == '-1':
        return JsonResponse({'retCode': '1000', 'message': '参数错误'})
    #app_id = 0
    has_training_model = ModelInfo.objects.filter(is_training=1,is_online=0)
    if has_training_model:
        return JsonResponse({'retCode': '1001', 'retDesc': '模型不可训练', 'isTrainAvailable': False})
    else:
        # train_thread=Thread(target=train_model, args=(app_id,))
        # train_thread.start()
        start_time = datetime.datetime.today()
        # 获取训练数据
        para = {'appId': app_id}
        get_data = requests.get('http://182.92.4.200:3210/AIaspect/trainAI',params=para)#appId!!!!
        if get_data.status_code != requests.codes.ok:
            return JsonResponse({'retCode': '1002', 'retDesc': '知识库请求失败'})
        retDesc = get_data.json()
        if retDesc.has_key('resQuestion'):
            retDesc=retDesc['resQuestion']
            nb_class = retDesc[1][0]['numClass']
            retDesc = retDesc[0]
        else:
            return JsonResponse({'retCode':'1003','retDesc':'模型获得数据量为0'})

        classify_m_info = ModelInfo.objects.filter(app_id=app_id,is_online=0)
        ModelInfo.objects.filter(id=classify_m_info[0].id).update(is_training=1)

        classify_train_cache_path = classify_m_info[0].offline_url[:-3] + 'train.h5'
        with open(BASE_DIR+'word_index.pkl', 'rb') as vocab:
            word_index = pickle.load(vocab)

        user_question = []
        classify_label = []
        for aDesc in retDesc:
            user_question.append(aDesc['question'])
            classify_label.append(aDesc['tid'])
        user_question_vec = []  # index后的用户问题
        for question in user_question:
            question_vec = []
            for word in cut_unchinese(question):
                if word in word_index:
                    question_vec.append(word_index[word])
                else:
                    question_vec.append(0)
            user_question_vec.append(question_vec)

        data = pad_sequences(user_question_vec, maxlen=classify_sequence_length)

        labels =to_categorical(np.asarray(classify_label),nb_class)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(0.2 * data.shape[0])
        x_train = data[:-2 * nb_validation_samples]
        y_train = labels[:-2 * nb_validation_samples]
        x_val = data[-2 * nb_validation_samples:-1 * nb_validation_samples]
        y_val = labels[-2 * nb_validation_samples:-1 * nb_validation_samples]
        x_test = data[-1 * nb_validation_samples:]
        y_test = labels[-1 * nb_validation_samples:]

        # 获取old model在测试集精度和损失
        classify_model = load_model(classify_m_info[0].offline_url)
        classify_score_old = classify_model.evaluate(x_test, y_test)

        # 训练模型
        sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
        classify_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath=classify_train_cache_path, save_best_only=True, verbose=1)
        classify_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nb_epoch, batch_size=batch_size,
                           shuffle=True, callbacks=[checkpointer, earlyStopping])

        # 获取train model在测试集精度和损失
        classify_model.load_weights(classify_train_cache_path)
        classify_score_train = classify_model.evaluate(x_test, y_test)

        cost_secends = (datetime.datetime.today() - start_time).total_seconds()
        cost_time = '%d:%d:%d' % (cost_secends / 3600, (cost_secends % 3600) / 60, cost_secends % 60)
        classify_m_history = ModelHistory.objects.filter(model_id=classify_m_info[0].id).order_by('-train_start_time')
        ModelHistory.objects.create(model_id=classify_m_info[0].id, train_data_num=len(data),
                                    accuracy=classify_score_train[1], loss=classify_score_train[0],
                                    train_start_time=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    train_cost_time=cost_time)
        length = len(classify_m_history)
        if length > 0:
            if classify_score_old[1] < classify_score_train[1]:
                os.remove(classify_m_info[0].offline_url)
                os.rename(classify_train_cache_path, classify_m_info[0].offline_url)
                ####################
                trans_model(classify_m_info[0].offline_url, app_id)
                print("发送模型") 
                ModelInfo.objects.filter(app_id=app_id, is_online=1).update(online_url=online_path+'classify'+str(app_id)+".h5",is_replace=1)                    
            else:
                os.remove(classify_train_cache_path)
        else:
            os.remove(classify_m_info[0].offline_url)
            os.rename(classify_train_cache_path, classify_m_info[0].offline_url)

        ModelInfo.objects.filter(id=classify_m_info[0].id).update(is_training=0)
        return JsonResponse({'retCode': '0', 'retDesc': '模型已训练完毕'})

def total_train(request):
    app_id = request.GET.get('appId', '-1')
    if app_id == '-1':
        return JsonResponse({'retCode': '1000', 'message': '参数错误'})
    #app_id = 0
    has_training_model = ModelInfo.objects.filter(is_training=1,is_online=0)
    if has_training_model:
        return JsonResponse({'retCode': '1001', 'retDesc': '模型不可训练', 'isTrainAvailable': False})
    else:
        start_time = datetime.datetime.today()
        # 获取训练数据
        para = {'appId': app_id}
        get_data = requests.get('http://182.92.4.200:3210/AIaspect/getAllQuestionsFromQdb',params=para)#appId!!!!
        if get_data.status_code != requests.codes.ok:
            return JsonResponse({'retCode': '1002', 'retDesc': '知识库请求失败'})
        retDesc = get_data.json()
        if retDesc.has_key('question'):
            retDesc=retDesc['question']
            # nb_class = retDesc[1][0]['numClass']
            # retDesc = retDesc[0]
        else:
            return JsonResponse({'retCode':'1003','retDesc':'模型获得数据量为0'})

        classify_m_info = ModelInfo.objects.filter(app_id=app_id,is_online=0)
        ModelInfo.objects.filter(id=classify_m_info[0].id).update(is_training=1)

        classify_train_cache_path = classify_m_info[0].offline_url[:-3] + 'train.h5'
        with open(BASE_DIR+'word_index.pkl', 'rb') as vocab:
            word_index = pickle.load(vocab)

        user_question = []
        classify_label = []
        for aDesc in retDesc:
            user_question.append(aDesc['question'])
            classify_label.append(aDesc['tid'])
        user_question_vec = []  # index后的用户问题
        for question in user_question:
            question_vec = []
            for word in cut_unchinese(question):
                if word in word_index:
                    question_vec.append(word_index[word])
                else:
                    question_vec.append(0)
            user_question_vec.append(question_vec)

        data = pad_sequences(user_question_vec, maxlen=classify_sequence_length)

        labels =to_categorical(np.asarray(classify_label),31)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(0.2 * data.shape[0])
        x_train = data[:-2 * nb_validation_samples]
        y_train = labels[:-2 * nb_validation_samples]
        x_val = data[-2 * nb_validation_samples:-1 * nb_validation_samples]
        y_val = labels[-2 * nb_validation_samples:-1 * nb_validation_samples]
        x_test = data[-1 * nb_validation_samples:]
        y_test = labels[-1 * nb_validation_samples:]

        # 获取old model在测试集精度和损失
        classify_model = load_model(classify_m_info[0].offline_url)
        classify_score_old = classify_model.evaluate(x_test, y_test)

        # 训练模型
        sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
        classify_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath=classify_train_cache_path, save_best_only=True, verbose=1)
        classify_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nb_epoch, batch_size=batch_size,
                           shuffle=True, callbacks=[checkpointer, earlyStopping])

        # 获取train model在测试集精度和损失
        classify_model.load_weights(classify_train_cache_path)
        classify_score_train = classify_model.evaluate(x_test, y_test)

        cost_secends = (datetime.datetime.today() - start_time).total_seconds()
        cost_time = '%d:%d:%d' % (cost_secends / 3600, (cost_secends % 3600) / 60, cost_secends % 60)
        classify_m_history = ModelHistory.objects.filter(model_id=classify_m_info[0].id).order_by('-train_start_time')
        ModelHistory.objects.create(model_id=classify_m_info[0].id, train_data_num=len(data),
                                    accuracy=classify_score_train[1], loss=classify_score_train[0],
                                    train_start_time=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    train_cost_time=cost_time)
        length = len(classify_m_history)
        if length > 0:
            if classify_score_old[1] < classify_score_train[1]:
                os.remove(classify_m_info[0].offline_url)
                os.rename(classify_train_cache_path, classify_m_info[0].offline_url)
                ####################
                trans_model(classify_m_info[0].offline_url, app_id)
                print("发送模型")
                ModelInfo.objects.filter(app_id=app_id, is_online=1).update(online_url=online_path+'classify'+str(app_id)+"1.h5",is_replace=1)
            else:
                os.remove(classify_train_cache_path)
        else:
            os.remove(classify_m_info[0].offline_url)
            os.rename(classify_train_cache_path, classify_m_info[0].offline_url)

        ModelInfo.objects.filter(id=classify_m_info[0].id).update(is_training=0)
        return JsonResponse({'retCode': '0', 'retDesc': '模型已训练完毕'})

def trans_model(local_path,appId):
    ip="182.92.222.75"
    port=22
    username='root'
    password='root!@#456'

    remote_path='/home/bothelper/BotHelper/'+"classify"+str(appId)+".h5"

    trans = paramiko.Transport((ip,port))
    trans.connect(username=username,password=password)

    sftp = paramiko.SFTPClient.from_transport(trans)


    sftp.put(localpath=local_path,remotepath=remote_path)

    trans.close()
    print("ftp over")

def cold_start(request):
    # cs_thread = Thread(target=loadGPU)
    # cs_thread.start()
    m_info = ModelInfo.objects.filter()
    model = load_model(m_info[0].offline_url)
    return JsonResponse({'retDesc': '冷启动完成'})
def loadGPU():
    m_info = ModelInfo.objects.filter()
    model = load_model(m_info[0].offline_url)
    del model

def train_model(app_id):
    start_time = datetime.datetime.today()
    classify_m_info = ModelInfo.objects.filter(app_id=app_id, type=0)
    #ModelInfo.objects.filter(id=classify_m_info[0].id).update(is_training=1)

    classify_train_cache_path = classify_m_info[0].offline_url[:-3] + 'train.h5'
    with open(BASE_DIR+'word_index.pkl', 'rb') as vocab:
        word_index = pickle.load(vocab)

    #获取训练数据
    get_data = requests.get('http://111.207.243.66:8880/QuestionBase/conversation/data')
    retDesc = get_data.json()['retDesc']
    user_question = []
    classify_label = []
    for aDesc in retDesc:
        user_question.append(aDesc['question'])
        classify_label.append(aDesc['firstCategoryId'])
    user_question_vec = []  # index后的用户问题
    for question in user_question:
        question_vec = []
        for word in cut_unchinese(question):
            if word in word_index:
                question_vec.append(word_index[word])
            else:
                question_vec.append(0)
        user_question_vec.append(question_vec)
    data = pad_sequences(user_question_vec, maxlen=classify_sequence_length)
    labels = to_categorical(np.asarray(classify_label))

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(0.2 * data.shape[0])
    x_train = data[:-2 * nb_validation_samples]
    y_train = labels[:-2 * nb_validation_samples]
    x_val = data[-2 * nb_validation_samples:-1 * nb_validation_samples]
    y_val = labels[-2 * nb_validation_samples:-1 * nb_validation_samples]
    x_test = data[-1 * nb_validation_samples:]
    y_test = labels[-1 * nb_validation_samples:]

    #获取old model在测试集精度和损失
    classify_model = load_model(classify_m_info[0].offline_url)
    classify_score_old = classify_model.evaluate(x_test, y_test)

    #训练模型
    sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    classify_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath=classify_train_cache_path,save_best_only=True,verbose=1)
    classify_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nb_epoch, batch_size=batch_size,
              shuffle=True, callbacks=[checkpointer, earlyStopping])

    #获取train model在测试集精度和损失
    classify_model.load_weights(classify_train_cache_path)
    classify_score_train = classify_model.evaluate(x_test, y_test)

    cost_secends = (datetime.datetime.today() - start_time).total_seconds()
    cost_time = '%d:%d:%d'%(cost_secends/3600,(cost_secends%3600)/60,cost_secends%60)
    classify_m_history = ModelHistory.objects.filter(model_id=classify_m_info[0].id).order_by('-train_start_time')
    ModelHistory.objects.create(model_id=classify_m_info[0].id, train_data_num=len(data), accuracy=classify_score_train[1], loss=classify_score_train[0],
                                train_start_time=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                                train_cost_time=cost_time)

    if len(classify_m_history)>0:
        if classify_score_old[1]<classify_score_train[1]:
            os.remove(classify_m_info[0].offline_url)
            os.rename(classify_train_cache_path,classify_m_info[0].offline_url)
        else:
            os.remove(classify_train_cache_path)
    else:
        os.remove(classify_m_info[0].offline_url)
        os.rename(classify_train_cache_path, classify_m_info[0].offline_url)

    #ModelInfo.objects.filter(id=classify_m_info[0].id).update(is_training=0)




