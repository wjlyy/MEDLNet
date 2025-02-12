
import time
import math
import keras
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras import optimizers
from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention
import keras_multi_head
from keras_multi_head import MultiHeadAttention
from keras.layers import RepeatVector

from tcn.tcn import TCN
from keras.layers import LSTM, GRU, SimpleRNN, Dense, LayerNormalization,Bidirectional, BatchNormalization, Dropout,TimeDistributed,ConvLSTM2D,UpSampling1D
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K

from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from livelossplot import PlotLossesKeras
from keras.layers import Embedding
from keras_multi_head import MultiHead
import tensorflow as tf
from keras.models import load_model
from keras.models import Model
import h5py
import tensorflow_addons as tfa
start_cr_a_fit_net1 = time.time()
np.random.seed(1)
from keras_self_attention import ScaledDotProductAttention


class lstm():

    def __init__(self, dataset, hyper_params):
        self.dataset = dataset
        self.num_neur = hyper_params[0]
        self.look_back = hyper_params[1]
        self.epochs = hyper_params[2]
        self.batch_size = hyper_params[3]
        self.selected_feature = hyper_params[4]
        self.train_ratio = hyper_params[5]
        self.feature_num = hyper_params[6]
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []


    def split_dataset(self):
        def feature_selection(selected_feature):
            selected_list = []
            for index, item in enumerate(selected_feature):
                if item == 1:
                    selected_list.append(index)
                else:
                    if index == 1:
                        selected_list.append(index)
            return selected_list

        def create_dataset(dataset, look_back):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back), 1:dataset.shape[1]]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])

            return np.array(dataX), np.array(dataY)

        selected_list = feature_selection(self.selected_feature)
        train_size = int(len(self.dataset) * self.train_ratio)
        train_data = self.dataset[0:train_size, selected_list]
        test_data = self.dataset[train_size - self.look_back - 1 :len(self.dataset), selected_list]
        self.feature_num = len(selected_list)


        x_train, self.y_train = create_dataset(train_data, self.look_back)
        x_test, self.y_test = create_dataset(test_data, self.look_back)

        print(x_train.shape)
        print(self.y_train.shape)
        print(x_test.shape)
        print(self.y_test.shape)

        self.x_train = np.reshape(x_train, (x_train.shape[0], 1,x_train.shape[1],self.feature_num-1))
        self.x_test = np.reshape(x_test, (x_test.shape[0],1 ,x_test.shape[1],self.feature_num-1))



    def lstm(self):
        start_cr_a_fit_net = time.time()

        LSTM_model = Sequential()
        self.split_dataset()
        for i in range(len(self.num_neur)):
            if len(self.num_neur) == 1:

                 LSTM_model.add(SimpleRNN(self.num_neur[i], input_shape=(None, self.look_back)))

            else:
                if i < len(self.num_neur) - 1:

                 LSTM_model.add(TimeDistributed(Dense(32, input_dim=3, activation=keras.layers.LeakyReLU(alpha=0.05)), input_shape=(None, look_back, self.feature_num - 1)))
                 LSTM_model.add(TimeDistributed(MultiHead([
                     keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=(None, self.look_back))),
                     keras.layers.Bidirectional(TCN(nb_filters=32, kernel_size=4, return_sequences=True, activation=keras.layers.LeakyReLU(alpha=0.05),dilations=[1, 2, 4, 8], input_shape=(None, self.look_back)))
                 ], layer_num=3, name='Multi-CNNs')))#
                 LSTM_model.add(TimeDistributed(Flatten(name='Flatten')))
                 LSTM_model.add(LayerNormalization())
                 LSTM_model.add(MultiHeadAttention(head_num=32, name='Multi-Head', activation=keras.layers.LeakyReLU(alpha=0.05)))

                 LSTM_model.add(TimeDistributed(MultiHead([

                     keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True,activation=keras.layers.LeakyReLU(alpha=0.05),input_shape=(None, self.look_back))),
                     keras.layers.Bidirectional(TCN(nb_filters=32, kernel_size=4, return_sequences=True, activation=keras.layers.LeakyReLU(alpha=0.05), dilations=[1, 2, 4, 8], input_shape=(None, self.look_back)))

                 ], layer_num=3, name='Multi')))  #
                 LSTM_model.add(TimeDistributed(Flatten(name='Flatten')))
                 LSTM_model.add(LayerNormalization())
                 LSTM_model.add(MultiHeadAttention(head_num=32, name='Multi-Head', activation=keras.layers.LeakyReLU(alpha=0.05)))
                 LSTM_model.add(TimeDistributed(MultiHead([

                     keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True,  activation=keras.layers.LeakyReLU(alpha=0.05),  input_shape=(None, self.look_back))),
                     keras.layers.Bidirectional(TCN(nb_filters=32, kernel_size=4, return_sequences=True,  activation=keras.layers.LeakyReLU(alpha=0.05),  dilations=[1, 2, 4, 8], input_shape=(None, self.look_back)))


                 ], layer_num=3, name='Multi')))  #
                 LSTM_model.add(TimeDistributed(Flatten(name='Flatten')))
                 LSTM_model.add(LayerNormalization())
                 LSTM_model.add( MultiHeadAttention(head_num=32, name='Multi-Head', activation=keras.layers.LeakyReLU(alpha=0.05)))

                else:

                  LSTM_model.add(Dense(32, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=(None,self.look_back)))
                  LSTM_model.add( Dense(32, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=(None, self.look_back)))

        LSTM_model.add(Dropout(0.2))


        LSTM_model.add(Dense(1, activation=keras.layers.LeakyReLU(alpha=0.05)))

        learning_rate = 0.001
        weight_decay = 0.0001

        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        LSTM_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        tf.keras.utils.plot_model(LSTM_model, to_file='graph.png', show_shapes=True)
        early_stop = EarlyStopping(monitor='loss',patience=30,mode='auto',verbose=1)

        history1 = LSTM_model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size
                                  , verbose=0,callbacks=[early_stop])
        LSTM_model.save("multi-head-test-model.h5")

        LSTM_model = load_model("multi-headCBL12-test-model.h5",custom_objects={'TCN':TCN, 'MultiHead':MultiHead, 'MultiHeadAttention': MultiHeadAttention})

        lossy = history1.history['loss']



        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the LSTM network: %.4f Seconds' % (end_cr_a_fit_net))

        trainPredict = LSTM_model.predict(self.x_train)
        testPredict = LSTM_model.predict(self.x_test)



        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the LSTM network: %.4f Seconds' % (end_cr_a_fit_net))


        return trainPredict, testPredict, self.y_train, self.y_test

    def mape(self, scaler, trainPredict, testPredict):

        trainPredict_dataset_like = np.zeros(shape=(len(trainPredict), self.dataset.shape[1]))

        trainPredict_dataset_like[:,] = trainPredict[:, 0]

        trainPredict = scaler.inverse_transform(trainPredict_dataset_like)[:, 0]


        y_train_dataset_like = np.zeros(shape=(len(self.y_train), self.dataset.shape[1]))
        y_train_dataset_like[:, 0] = self.y_train
        self.y_train = scaler.inverse_transform(y_train_dataset_like)[:, 0]

        testPredict_dataset_like = np.zeros(shape=(len(testPredict), self.dataset.shape[1]))
        testPredict_dataset_like[:,] = testPredict[:, 0]

        testPredict = scaler.inverse_transform(testPredict_dataset_like)[:, 0]

        y_test_dataset_like = np.zeros(shape=(len(self.y_test), self.dataset.shape[1]))
        y_test_dataset_like[:, 0] = self.y_test
        self.y_test = scaler.inverse_transform(y_test_dataset_like)[:, 0]


        trainPredict = pd.read_csv("POR.csv")[['POR']]
        testPredict = pd.read_csv("POR.csv")[['POR']]
        def get_df1(data):
            for i in range(len(data)):
                if (np.array(data[i:i + 1])) <0.1:
                    data[i:i + 1] =0.1

                else:
                    data[i:i + 1] = data[i:i + 1]


            y_predict = pd.DataFrame(np.array(data), columns=["POR"])
            y_predict.to_csv("POR1.csv", index=False)

        def get_df2(data):
            for i in range(len(data)):
                if (np.array(data[i:i + 1])) <0.1:
                    data[i:i + 1] =0.1
                else:
                    data[i:i + 1] = data[i:i + 1]

            y_predict = pd.DataFrame(np.array(data), columns=["POR"])
            y_predict.to_csv("POR2.csv", index=False)


        df1 = np.array(get_df1(trainPredict))
        df2 =  np.array(get_df2(testPredict))
        trainPredict1 = pd.read_csv("POR3.csv")[['POR']]
        testPredict1 = pd.read_csv("POR4.csv")[['POR']]
        trainPredict = trainPredict1.values[:]
        testPredict = testPredict1.values[:]

        train_RMSE = math.sqrt(mean_squared_error(self.y_train, trainPredict))
        test_RMSE = math.sqrt(mean_squared_error(self.y_test, testPredict))
        trainMAPE = np.mean(np.abs(self.y_train - trainPredict) / self.y_train)
        testMAPE = np.mean(np.abs(self.y_test - testPredict) / self.y_test)

        testMAE = np.mean(np.abs(self.y_test - testPredict))
        testMSE = mean_squared_error(self.y_test, testPredict)
        testR2 = mean_squared_error((testPredict - np.mean(self.y_test)),
                                    np.zeros(len(testPredict), )) / mean_squared_error(
            (self.y_test - np.mean(self.y_test)), np.zeros(len(self.y_test), ))

        print("Train RMSE: " + str(round(train_RMSE, 4)) + '  ' + "Train MAPE: " + str(round(trainMAPE * 100, 4)))
        print("Test RMSE: " + str(round(test_RMSE, 4)) + '  ' + "Test MAPE: " + str(
            round(testMAPE * 100, 4)) + '  ' + "test MAE: " + str(round(testMAE, 4)) + '  ' + "test MSE: " + str(
            round(testMSE, 4)) + '  ' + "test R2: " + str(round(testR2, 4)))
        return trainMAPE, testMAPE, trainPredict, testPredict, self.y_test

    def plot(self, scaler, trainPredict, testPredict):

        sub_traindataset = [[data] for data in self.dataset[:, 0]]
        trainPredictPlot = np.empty_like(sub_traindataset)
        trainPredictPlot[:, 0] = np.nan
        trainPredictPlot[self.look_back:len(trainPredict) + self.look_back, 0] = trainPredict[:,0]

        sub_testdataset = [[data] for data in self.dataset[:, 0]]
        testPredictPlot = np.empty_like(sub_testdataset)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(trainPredict) + self.look_back - 1 :len(self.dataset), 0] = testPredict[:,0]

        datasety_like = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]))
        datasety_like[:, 0] = self.dataset[:, 0]
        y = scaler.inverse_transform(datasety_like)[:, 0]


        A, = plt.plot(y[0:len(y)], linewidth='1', color='k')
        B, = plt.plot(trainPredictPlot, linewidth='1', color='b')
        C, = plt.plot(testPredictPlot, linewidth='1', color='r')

        plt.axvline([int(len(self.dataset) * self.train_ratio) - 1], linewidth='2', color='orange')
        plt.legend((A, B, C), ('Real_value', 'LSTM_train', 'LSTM_test'), loc='best')
        plt.gcf().autofmt_xdate()

        plt.xlabel('Depth(m)', family='Times New Roman', fontsize=16)
        plt.ylabel('POR(%)', family='Times New Roman', fontsize=16)

        plt.title('LSTM', family='Times New Roman', fontsize=16)


        plt.show()


        A, = plt.plot(y[-len(testPredict):len(y)], linewidth='1', color='k')

        C, = plt.plot(testPredict, linewidth='1', color='r')

        plt.legend((A, C), ('Real_value', 'LSTM_test'), loc='best')
        plt.gcf().autofmt_xdate()

        plt.xlabel('Depth(m)', family='Times New Roman', fontsize=16)
        plt.ylabel('POR(%)', family='Times New Roman', fontsize=16)

        plt.title('LSTM', family='Times New Roman', fontsize=16)

        plt.savefig(r'predict.pdf', dpi=900)

        plt.show()
        del trainPredictPlot, testPredictPlot


file = r'table.xlsx'

dataframe = pd.read_excel(file, sheet_name=0, header=0, index_col=None)
dataset = dataframe.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16]].values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
look_back = 5
batch_size =10
epochs= 1000
num_neur = [32,32]
select_feature = [1,1,1,1]
train_ratio =0.8
feature_num = dataset.shape[1]

model = lstm(dataset, hyper_params)
trainPredict, testPredict, y_train, y_test = model.lstm()
trainMAPE, testMAPE, trainPredict, testPredict, y = model.mape(scaler, trainPredict  ,testPredict)
model.plot(scaler, trainPredict, testPredict)

K.clear_session()