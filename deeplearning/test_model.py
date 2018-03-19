import numpy as np
import sys
import os
import tensorflow as tf
import keras as keras
import h5py
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Activation,Dense,Dropout,Embedding,Flatten,Input,merge,Convolution2D,MaxPooling2D,Convolution1D,MaxPooling1D,merge,Reshape,concatenate
from keras.optimizers import SGD,sgd
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from gensim.models import Word2Vec
from keras import regularizers

reload(sys)
sys.setdefaultencoding('utf-8')
classify_sequence_length = 100
sequence_length = 100
embedding_dim=100
num_epoch_set = 10
batch_size_set=32
text_paths=["/home/wxr/blazer/YiQiZuoYe/corpus/train",]

#Data Preproccesing
def load_data():
    #load data
    print("loading data...")
    texts = []  # list of text samples
    labels = []  # list of label ids
    labels_3 = []
    count = -1
    for path_name in text_paths:
        for name in sorted(os.listdir(path_name)):
            print "name : ",name
            if(name[0]!='.'):
                count += 1
                print count
                namee = name.decode('utf-8')
                fpath = os.path.join(path_name,namee)
                #print fpath
                for name_2 in sorted(os.listdir(fpath)):
                    if(name_2[0]!='.'):
                        namee_2 = name_2.decode('utf-8')
                        fpath_2 = os.path.join(fpath, name_2)
                        #print fpath_2
                        for real_name in sorted(os.listdir(fpath_2)):
                            if(real_name[0]!='.'):
                                #real_namee = real_name.decode('utf-8')
                                real_namee = real_name
                                fpath3 = os.path.join(path_name,namee,namee_2,real_namee)
                                #print fpath3
                                f=open(fpath3,'rU')
                                text=f.read()
                                #print text
                                texts.append(text)
                                f.close()
                                class_3 = namee_2
                                #print "add label : ",intname
                                labels.append(count)
                                labels_3.append(class_3)

    
    sequences = []
    for text in texts:
        num = 0
        sequence = []
        text_split = text.split(" ")
        for word in text_split:
            if word!="" and num <= sequence_length:
                num += 1
                word = word.decode("utf-8")
                if word in charvec_model.vocab:
                    sequence.append(charvec_model.vocab[word].index + 1)
                    #print "coming"
                else:
                    #print "no"
                    sequence.append(0)
        sequences.append(sequence)

    #padding the sequences to the same length
    data = pad_sequences(sequences, maxlen = sequence_length)
    
    
    labels = to_categorical(np.asarray(labels))
    
    indices = np.arange(data.shape[0])
    #print "indices_shape : ",indices
    np.random.shuffle(indices)
    #print "random indices : ",indices
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(0.2 * data.shape[0])

    x_train = data[:-1*nb_validation_samples]
    y_train = labels[:-1*nb_validation_samples]
    x_val = data[-1*nb_validation_samples:]
    y_val = labels[-1*nb_validation_samples:]
    


    return x_train, y_train, x_val, y_val

if __name__ == "__main__":
    #with tf.device('/gpu:0'):
        ##############
        #load data
        ##############
    charvec_model = Word2Vec.load('/home/wxr/hyx/Yqzy/yqzy_word100.model')
    x_train, y_train, x_valid, y_valid = load_data()


    nb_words = len(charvec_model.vocab)
    print "embedding:%d" % (nb_words + 1)
    embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
    embedding_matrix[1:, :] = charvec_model.syn0

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
    total_input = Input(shape=(100,))
    
    model1 = embedding_layer1(total_input)
    model1 = Reshape((classify_sequence_length, 100, 1))(model1)

    model2 = embedding_layer2(total_input)
    model2 = Reshape((classify_sequence_length, 100, 1))(model2)
    
    graph_in = concatenate(inputs=[model1,model2], axis=-1)
    conv_1 = keras.layers.convolutional.Conv2D(filters=128, kernel_size=(1,100), strides=(1, 1), padding='valid', activation='relu')(graph_in)
    conv_2 = keras.layers.convolutional.Conv2D(filters=128, kernel_size=(2,100), strides=(1, 1), padding='valid', activation='relu')(graph_in)
    conv_3 = keras.layers.convolutional.Conv2D(filters=128, kernel_size=(3,100), strides=(1, 1), padding='valid', activation='relu')(graph_in)
    
    
    conv_1 = MaxPooling2D(pool_size=(int(conv_1.get_shape()[1]), 1))(conv_1)
    conv_2 = MaxPooling2D(pool_size=(int(conv_2.get_shape()[1]), 1))(conv_2)
    conv_3 = MaxPooling2D(pool_size=(int(conv_3.get_shape()[1]), 1))(conv_3)
    conva = concatenate(inputs=[conv_1,conv_2,conv_3], axis=-1)

    out = Reshape((3 * 128,))(conva)
    out = keras.layers.core.Dense(units=60, activation='relu', kernel_regularizer=regularizers.l2(0.01))(out)
    out = Dropout(0.3)(out)
    out = keras.layers.core.Dense(units=5, activation='softmax')(out)
    total = Model(inputs=total_input, outputs=out)

    sgd1=keras.optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    total.compile(optimizer = sgd1, loss='categorical_crossentropy', metrics=["accuracy"])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath="/home/wxr/hyx/604testmodel.h5",verbose=1,
                                   save_best_only=True)
    total.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=num_epoch_set, batch_size=batch_size_set, shuffle=True,callbacks=[checkpointer, earlyStopping])
    total.load_weights('/home/wxr/hyx/604testmodel.h5')
    print"evaluate:"
    score = total.evaluate(x_train, y_train)
    print('Train loss:' , score[0])
    print('Train accuracy:', score[1])

    score = total.evaluate(x_valid, y_valid)
    print('validation loss:', score[0])
    print('validation accuracy:', score[1])

