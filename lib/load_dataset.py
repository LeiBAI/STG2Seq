import numpy as np
import pandas as pd
import h5py
import time
import os
from lib.normalization import *
from data.TaxiBJ.load_dataset_BJ import load_BJ_dataset

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

def extract_hour(strings):
    #strings is a list contains a set of time stamps
    hours = []
    for t in strings:
        year, month, day, hour = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        hours.append(hour)
    hours = np.array(hours)[:, np.newaxis]
    return hours


def extract_weekday(strings):
    weekdays = []
    for t in strings:
        vec = time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday
        weekdays.append(vec)
    weekdays = np.array(weekdays)[:, np.newaxis]
    return weekdays

def load_SY_data(base_dir, nb_flow, height, width):
    # shape of demand_demand in SY is (len, H*W)
    demand_data = np.loadtxt(os.path.join(base_dir, 'data/Didi_SY/didi_order_hour_grid.csv'),
                             delimiter=',', skiprows=1)
    scaler = MinMaxScaler(demand_data.min(), demand_data.max())
    demand_data = scaler.transform(demand_data)
    len_data = demand_data.shape[0]
    demand_data = np.reshape(demand_data, (len_data, nb_flow, height, width))
    external_data = pd.read_csv(os.path.join(base_dir, 'data/Didi_SY/externalFeature.csv'), header=0)

    hours = external_data['TimeOfDay'].values[:, np.newaxis]
    weekdays = external_data['TimeOfWeek'].values[:, np.newaxis]
    holidays = external_data['holiday'].values[:, np.newaxis]
    external_time = np.hstack((hours, weekdays, holidays))
    external_time = one_hot_by_column(external_time)
    print('Load External Time Data:', external_time.shape)

    weather_state = external_data['weatherState'].values[:, np.newaxis]
    weather_state = one_hot_by_column(weather_state)
    successive_features = external_data[['temperature', 'windSpeed', 'visibility']]
    successive_features = minmax_by_column(successive_features.values)
    external_weather = np.hstack((weather_state, successive_features))
    print("Load External Weather Data:", external_weather.shape)

    external = np.hstack((external_time, external_weather))
    return demand_data, external, scaler, len_data

def load_SY_data_irregular(base_dir, nb_flow, height, width):
    # shape of demand_demand in SY is (len, H*W)
    demand_data = np.loadtxt(os.path.join(base_dir, 'data/Didi_SY/didi_order_hour.csv'),
                             delimiter=',', skiprows=1)
    scaler = MinMaxScaler(demand_data.min(), demand_data.max())
    demand_data = scaler.transform(demand_data)
    len_data = demand_data.shape[0]
    demand_data = np.reshape(demand_data, (len_data, nb_flow, height, width))
    external_data = pd.read_csv(os.path.join(base_dir, 'data/Didi_SY/externalFeature.csv'), header=0)

    hours = external_data['TimeOfDay'].values[:, np.newaxis]
    weekdays = external_data['TimeOfWeek'].values[:, np.newaxis]
    holidays = external_data['holiday'].values[:, np.newaxis]
    external_time = np.hstack((hours, weekdays, holidays))
    external_time = one_hot_by_column(external_time)
    print('Load External Time Data:', external_time.shape)

    weather_state = external_data['weatherState'].values[:, np.newaxis]
    weather_state = one_hot_by_column(weather_state)
    successive_features = external_data[['temperature', 'windSpeed', 'visibility']]
    successive_features = minmax_by_column(successive_features.values)
    external_weather = np.hstack((weather_state, successive_features))
    print("Load External Weather Data:", external_weather.shape)

    external = np.hstack((external_time, external_weather))
    return demand_data, external, scaler, len_data

def load_SYDNEY_data(base_dir, nb_flow, height, width):
    # shape of demand_demand in SY is (len, H*W)
    demand_data = np.loadtxt(os.path.join(base_dir, 'data/1h_data_use.csv'),
                             delimiter=',', skiprows=1)
    scaler = MinMaxScaler(demand_data.min(), demand_data.max())
    demand_data = scaler.transform(demand_data)
    len_data = demand_data.shape[0]
    demand_data = np.reshape(demand_data, (len_data, nb_flow, height, width))
    external = np.zeros((len_data, 10))
    return demand_data, external, scaler, len_data

def load_NYC_data(base_dir, nb_flow, height, width):
    # shape of demand_data in NYC is (len, 2, H, W)
    demand_data, timestamps = load_stdata(os.path.join(base_dir, 'data/BikeNYC/NYC14_M16x8_T60_NewEnd.h5'))
    demand_data = demand_data[:, :nb_flow]
    scaler = MinMaxScaler(demand_data.min(), demand_data.max())
    demand_data = scaler.transform(demand_data)
    len_data = demand_data.shape[0]
    hours = extract_hour(timestamps)
    weekdays = extract_weekday(timestamps)
    external_time = np.hstack((hours, weekdays))
    external_time = one_hot_by_column(external_time)
    print('Load External Time Data:', external_time.shape)

    external = external_time
    return demand_data, external, scaler, len_data




def extract_closeness(source, index, end_index, closeness):
    len = source.shape[0]
    results = []
    while index < end_index:
        depends = [index - i for i in range(1, closeness + 1)]
        result = source[depends]
        results.append(result)
        index = index + 1
    return results

def extract_period(source, index, end_index, period, T=24):
    len = source.shape[0]
    results = []
    while index < end_index:
        depends = [index - i*T for i in range(1, period + 1)]
        result = source[depends]
        results.append(result)
        index = index + 1
    return results

def extract_trend(source, index, end_index, trend, T=24):
    len = source.shape[0]
    results = []
    while index < end_index:
        depends = [index - i*T*7 for i in range(1, trend + 1)]
        result = source[depends]
        results.append(result)
        index = index + 1
    return results

def extract_external(source, index, end_index, seq):
    len = source.shape[0]
    results = []
    while index < end_index:
        depends = [index - i for i in range(1, seq + 1)]
        result = source[depends]
        results.append(result)
        index = index + 1
    return results

def load_dataset(base_dir, dataset, nb_flow, height, width, T=24,
                 len_closeness=4, len_period=3, len_trend=0, test_days = 6, external_dim=0):
    # load dataset from location base_dir
    # this model return label data without horizon
    # X is a list consisted of closeness, period, trend, external
    # Y is tensor
    if dataset == 'SY':
        demand, external, scaler, len = load_SY_data(base_dir, nb_flow, height, width)
    elif dataset == 'SY_IR':
        demand, external, scaler, len = load_SY_data_irregular(base_dir, nb_flow, height, width)
    elif dataset == 'NYC':
        demand, external, scaler, len = load_NYC_data(base_dir, nb_flow, height, width)
    elif dataset == 'BJ':
        demand, external, scaler, len = load_BJ_dataset(base_dir)
    else:
        print('Dataset Error!')
        exit(0)
    start_index = max(7 * T * len_trend, T * len_period, len_closeness)

    index = start_index
    Y = []
    while index < len:
        Y.append(demand[index])
        index += 1
    Y = np.stack(Y)
    Y = np.transpose(Y, (0, 2, 3, 1))
    Y_train = Y[:-test_days * T]
    Y_test = Y[-test_days * T:]

    X_train = []
    X_test = []
    if len_closeness > 0:
        index = start_index
        XC = extract_closeness(demand, index, len, len_closeness)
        XC = np.stack(XC)
        #XC = np.reshape(XC, (XC.shape[0], -1, XC.shape[3], XC.shape[4]))
        XC = np.transpose(XC, (0, 1, 3, 4, 2))
        XC_train = XC[:-test_days * T]
        XC_test = XC[-test_days * T:]
        X_train.append(XC_train)
        X_test.append(XC_test)
    if len_period > 0:
        index = start_index
        XP = extract_period(demand, index, len, len_period, T=T)
        XP = np.stack(XP)
        #XP = np.reshape(XP, (XP.shape[0], -1, XP.shape[3], XP.shape[4]))
        XP = np.transpose(XP, (0, 1, 3, 4, 2))
        XP_train = XP[:-test_days * T]
        XP_test = XP[-test_days * T:]
        X_train.append(XP_train)
        X_test.append(XP_test)
    if len_trend > 0:
        index = start_index
        XT = extract_trend(demand, index, len, len_trend, T=T)
        XT = np.stack(XT)
        #XT = np.reshape(XT, (XT.shape[0], -1, XT.shape[3], XT.shape[4]))
        #XT = np.transpose(XT, (0, 2, 3, 1))
        XT = np.transpose(XT, (0, 1, 3, 4, 2))
        XT_train = XT[:-test_days * T]
        XT_test = XT[-test_days * T:]
        X_train.append(XT_train)
        X_test.append(XT_test)

    index = start_index
    if external_dim == 1:
        X_train.append(external[index:-test_days * T])
        X_test.append(external[-test_days * T:])
    if external_dim > 1:
        external = extract_external(external, index, len, len_closeness)
        external = np.stack(external)
        X_train.append(external[:-test_days * T])
        X_test.append(external[-test_days * T:])


    print('Dataset loaded!')
    return X_train, Y_train, X_test, Y_test, scaler


def load_dataset_with_horizon(base_dir, dataset, nb_flow, height, width, T =24,
                              len_closeness=4, len_period=3, len_trend=0, horizon=1,
                              test_days = 6, external_dim=0):

    #load dataset from location base_dir
    # this model return horizon data without external features
    # X is a list consisted of closeness, period, trend, external
    # Y is tensor: horizon
    if dataset == 'SY':
        demand, external, scaler, len = load_SY_data(base_dir, nb_flow, height, width)
    elif dataset == 'SY_IR':
        demand, external, scaler, len = load_SY_data_irregular(base_dir, nb_flow, height, width)
    elif dataset == 'NYC':
        demand, external, scaler, len = load_NYC_data(base_dir, nb_flow, height, width)
    elif dataset == 'BJ':
        demand, external, scaler, len = load_BJ_dataset(base_dir)
    else:
        print('Dataset Error!')
        exit(0)
    start_index = max(7 * T * len_trend, T * len_period, len_closeness)
    end_index = len-horizon+1

    index = start_index
    Y = []
    if horizon == 1:
        while index < end_index:
            Y.append(demand[index])
            index += 1
        Y = np.stack(Y)
        Y = np.transpose(Y, (0, 2, 3, 1))
        Y_train = Y[:-test_days * T]
        Y_test = Y[-test_days * T:]
    if horizon > 1:
        while index < end_index:
            Y.append(demand[index:index+horizon])
            index += 1
        Y = np.stack(Y)
        Y = np.transpose(Y, (0, 1, 3, 4, 2))
        Y_train = Y[:-test_days * T]
        Y_test = Y[-test_days * T:]

    X_train = []
    X_test = []
    if len_closeness > 0:
        index = start_index
        XC = extract_closeness(demand, index, end_index, len_closeness)
        XC = np.stack(XC)
        #XC = np.reshape(XC, (XC.shape[0], -1, XC.shape[3], XC.shape[4]))
        #after the transpose, shape is [N, C, H, W, nb_flow]
        XC = np.transpose(XC, (0, 1, 3, 4, 2))
        XC_train = XC[:-test_days * T]
        XC_test = XC[-test_days * T:]
        X_train.append(XC_train)
        X_test.append(XC_test)
    if len_period > 0:
        index = start_index
        XP = extract_period(demand, index, end_index, len_period, T=T)
        XP = np.stack(XP)
        #XP = np.reshape(XP, (XP.shape[0], -1, XP.shape[3], XP.shape[4]))
        XP = np.transpose(XP, (0, 1, 3, 4, 2))
        XP_train = XP[:-test_days * T]
        XP_test = XP[-test_days * T:]
        X_train.append(XP_train)
        X_test.append(XP_test)
    if len_trend > 0:
        index = start_index
        XT = extract_trend(demand, index, end_index, len_trend, T=T)
        XT = np.stack(XT)
        #XT = np.reshape(XT, (XT.shape[0], -1, XT.shape[3], XT.shape[4]))
        #XT = np.transpose(XT, (0, 2, 3, 1))
        XT = np.transpose(XT, (0, 1, 3, 4, 2))
        XT_train = XT[:-test_days * T]
        XT_test = XT[-test_days * T:]
        X_train.append(XT_train)
        X_test.append(XT_test)

    index = start_index
    if external_dim == 1:
        X_train.append(external[index:-test_days * T])
        X_test.append(external[-test_days * T-horizon+1: end_index])
    if external_dim > 1:
        external = extract_external(external, index, end_index, len_closeness)
        external = np.stack(external)
        X_train.append(external[:-test_days * T])
        X_test.append(external[-test_days * T:])


    print('Dataset loaded!')
    for i in X_train:
        print(i.shape)
    for i in X_test:
        print(i.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    return X_train, Y_train, X_test, Y_test, scaler

def load_dataset_multi_demand_external(base_dir, dataset, nb_flow, height, width, T=24,
                                       len_closeness=4, len_period=0, len_trend=0,
                                       test_days=6, horizon=1, external_length=0):

    #load dataset from location base_dir
    #this model return horizon data with external features
    #X is a list consisted of closeness, period, trend, external
    #Y is a list consisted of horizon, external
    if dataset == 'SY':
        demand, external, scaler, len = load_SY_data(base_dir, nb_flow, height, width)
    elif dataset == 'SY_IR':
        demand, external, scaler, len = load_SY_data_irregular(base_dir, nb_flow, height, width)
    elif dataset == 'NYC':
        demand, external, scaler, len = load_NYC_data(base_dir, nb_flow, height, width)
    elif dataset == 'BJ':
        demand, external, scaler, len = load_BJ_dataset(base_dir)
    elif dataset == 'SYDNEY':
        demand, external, scaler, len = load_SYDNEY_data(base_dir, nb_flow, height, width)
    else:
        print('Dataset Error!')
        exit(0)
    start_index = max(7 * T * len_trend, T * len_period, len_closeness)
    end_index = len-horizon+1

    index = start_index
    '''
    if horizon == 1:
        print('Horizon is 1, use another API!')
        exit(0)
    if horizon > 1:
    '''
    YD = []
    YE = []
    while index < end_index:
        YD.append(demand[index:index+horizon])
        YE.append(external[index:index+horizon])
        index += 1
    YD = np.stack(YD)
    YD = np.transpose(YD, (0, 1, 3, 4, 2))
    YD_train = YD[:-test_days * T]
    YD_test = YD[-test_days * T:]
    YE = np.stack(YE)
    YE_train = YE[:-test_days * T]
    YE_test = YE[-test_days * T:]
    Y_train = [YD_train, YE_train]
    Y_test = [YD_test, YE_test]

    X_train = []
    X_test = []
    if len_closeness > 0:
        index = start_index
        XC = extract_closeness(demand, index, end_index, len_closeness)
        XC = np.stack(XC)
        #XC = np.reshape(XC, (XC.shape[0], -1, XC.shape[3], XC.shape[4]))
        #after the transpose, shape is [N, C, H, W, nb_flow]
        XC = np.transpose(XC, (0, 1, 3, 4, 2))
        XC_train = XC[:-test_days * T]
        XC_test = XC[-test_days * T:]
        X_train.append(XC_train)
        X_test.append(XC_test)
    if len_period > 0:
        index = start_index
        XP = extract_period(demand, index, end_index, len_period, T=T)
        XP = np.stack(XP)
        #XP = np.reshape(XP, (XP.shape[0], -1, XP.shape[3], XP.shape[4]))
        XP = np.transpose(XP, (0, 1, 3, 4, 2))
        XP_train = XP[:-test_days * T]
        XP_test = XP[-test_days * T:]
        X_train.append(XP_train)
        X_test.append(XP_test)
    if len_trend > 0:
        index = start_index
        XT = extract_trend(demand, index, end_index, len_trend, T=T)
        XT = np.stack(XT)
        #XT = np.reshape(XT, (XT.shape[0], -1, XT.shape[3], XT.shape[4]))
        #XT = np.transpose(XT, (0, 2, 3, 1))
        XT = np.transpose(XT, (0, 1, 3, 4, 2))
        XT_train = XT[:-test_days * T]
        XT_test = XT[-test_days * T:]
        X_train.append(XT_train)
        X_test.append(XT_test)

    index = start_index
    if external_length == 1:
        X_train.append(external[index:-test_days * T -horizon+1])
        X_test.append(external[-test_days * T-horizon+1: end_index])
    if external_length > 1:
        external = extract_external(external, index, end_index, external_length)
        external = np.stack(external)
        X_train.append(external[:-test_days * T])
        X_test.append(external[-test_days * T:])


    print('Dataset loaded!')
    for i in X_train:
        print(i.shape)
    for i in X_test:
        print(i.shape)
    for i in Y_train:
        print(i.shape)
    for i in Y_test:
        print(i.shape)
    return X_train, Y_train, X_test, Y_test, scaler

def load_dataset_all_external(base_dir, dataset, nb_flow, height, width, T=24,
                                       len_closeness=4, len_period=0, len_trend=0,
                                       test_days=6, horizon=1):
    '''
    final version of the load dataset funciton, the funciton will load all demand, external(time+weather)
    for all historical data and data, time, weather for future data (horizon length). Users can
    select what they need in their code, instead of choosing data by parameters. So there is no
    need for external_length in previous version
    #X is a list consisted of closeness, period, trend, external
    #Y is a list consisted of horizon, external
    :param base_dir:
    :param dataset:
    :param nb_flow:
    :param height:
    :param width:
    :param T:
    :param len_closeness:
    :param len_period:
    :param len_trend:
    :param test_days:
    :param horizon:
    :return:
    '''
    if dataset == 'SY':
        #shape of demand is [N, nb_flow, height, width], external is [N, time+weather]
        demand, external, scaler, len = load_SY_data(base_dir, nb_flow, height, width)
    elif dataset == 'SY_IR':
        demand, external, scaler, len = load_SY_data_irregular(base_dir, nb_flow, height, width)
    elif dataset == 'NYC':
        demand, external, scaler, len = load_NYC_data(base_dir, nb_flow, height, width)
    elif dataset == 'BJ':
        demand, external, scaler, len = load_BJ_dataset(base_dir)
    else:
        print('Dataset Error!')
        exit(0)
    start_index = max(7 * T * len_trend, T * len_period, len_closeness)
    end_index = len - horizon + 1

    index = start_index
    YD = []
    YE = []
    while index < end_index:
        YD.append(demand[index:index + horizon])
        YE.append(external[index:index + horizon])
        index += 1
    YD = np.stack(YD)
    YD = np.transpose(YD, (0, 1, 3, 4, 2))
    YD_train = YD[:-test_days * T]
    YD_test = YD[-test_days * T:]
    YE = np.stack(YE)
    YE_train = YE[:-test_days * T]
    YE_test = YE[-test_days * T:]
    Y_train = [YD_train, YE_train]
    Y_test = [YD_test, YE_test]

    X_train = []
    X_test = []
    if len_closeness > 0:
        index = start_index
        XC = extract_closeness(demand, index, end_index, len_closeness)
        XC = np.stack(XC)
        # XC = np.reshape(XC, (XC.shape[0], -1, XC.shape[3], XC.shape[4]))
        # after the transpose, shape is [N, C, H, W, nb_flow]
        XC = np.transpose(XC, (0, 1, 3, 4, 2))
        XC_train = XC[:-test_days * T]
        XC_test = XC[-test_days * T:]
        X_train.append(XC_train)
        X_test.append(XC_test)
    if len_period > 0:
        index = start_index
        XP = extract_period(demand, index, end_index, len_period, T=T)
        XP = np.stack(XP)
        # XP = np.reshape(XP, (XP.shape[0], -1, XP.shape[3], XP.shape[4]))
        XP = np.transpose(XP, (0, 1, 3, 4, 2))
        XP_train = XP[:-test_days * T]
        XP_test = XP[-test_days * T:]
        X_train.append(XP_train)
        X_test.append(XP_test)
    if len_trend > 0:
        index = start_index
        XT = extract_trend(demand, index, end_index, len_trend, T=T)
        XT = np.stack(XT)
        # XT = np.reshape(XT, (XT.shape[0], -1, XT.shape[3], XT.shape[4]))
        # XT = np.transpose(XT, (0, 2, 3, 1))
        XT = np.transpose(XT, (0, 1, 3, 4, 2))
        XT_train = XT[:-test_days * T]
        XT_test = XT[-test_days * T:]
        X_train.append(XT_train)
        X_test.append(XT_test)

    index = start_index
    external = extract_external(external, index, end_index, len_closeness)
    external = np.stack(external)
    X_train.append(external[:-test_days * T])
    X_test.append(external[-test_days * T:])

    print('Dataset loaded!')
    for i in X_train:
        print(i.shape)
    for i in X_test:
        print(i.shape)
    for i in Y_train:
        print(i.shape)
    for i in Y_test:
        print(i.shape)
    return X_train, Y_train, X_test, Y_test, scaler
    pass

#example for use
if __name__ == '__main__':
    from RootPATH import base_dir
    from model.STGCN.params_stgcn import params_sy_ir as params
    base_dir = os.path.realpath(base_dir)

    X_train, Y_train, X_test, Y_test, scaler = load_dataset_all_external(
                                                            base_dir, params.source, params.nb_flow,
                                                            params.map_height, params.map_width,
                                                            len_closeness=12,
                                                            len_period=0,
                                                            len_trend=0,
                                                            test_days=6,
                                                            horizon=3)
    '''
    for i in X_train:
        print(i.shape)
    for i in X_test:
        print(i.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    '''

