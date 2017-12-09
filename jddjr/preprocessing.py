# encoding: utf-8
# author: huizhu
# created time: 2017年12月06日 星期三 19时02分26秒

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


DATA_DIR = os.path.join(os.path.expanduser('~'), 'datasets/test_data/jddjr/salesForecast')
NAME = ['comment', 'order', 'ads', 'product', 'sales_sum']
DATE = {'comment': 'create_dt', 
        'order': 'ord_dt', 
        'ads': 'create_dt', 
        'product': 'on_dt', 
        'sales_sum': 'dt'}

START = '2016-08-03'
END = '2017-04-30'


def generate_csv_by_shop():
    """
    将 5 个文件依据 shop_id 拆分为子文件，并按时间进行排序; 
    比如对于 t_comment.csv，会创建文件夹 comment/ 并在该文件夹下生成文件 
    shop_1.csv, shop_2.csv, ..., shop_3000.csv,　
    则 shop_i.csv 中仅包含该商店的 comment 数据
    """
    for n in NAME:
        df = pd.read_csv(os.path.join(DATA_DIR, 't_{0}.csv'.format(n)), parse_dates=[0])
        sub_dir = os.path.join(DATA_DIR, n)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        for ii in range(1, 3001):
            df_sub = df[df.shop_id == ii]
            df_sub = df_sub.sort_values(by=DATE[n])
            filename = os.path.join(sub_dir, 'shop_{0}.csv'.format(ii))
            df_sub.to_csv(filename, index=False)


def generate_sale_amt_by_day(shop_id_list):
    """
    因为商店每天的销售额数据会有多条, 本函数会将某个商店 (shop_id) 的销售额数据 (sale_amt) 
    依据 天 (2016-08-03 至 2017-04-03) 进行求和，将结果写入 sale_amt_by_day/shop_i.csv 中
    """
    output_dir = os.path.join(DATA_DIR, 'sale_amt_by_day')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for shop_id in shop_id_list:
        order_file = os.path.join(DATA_DIR, 'order/shop_{0}.csv'.format(shop_id))
        df_order = pd.read_csv(order_file, parse_dates=[0])
        df_order.index = df_order['ord_dt'].tolist()
        df_order = df_order.sort_values(by='ord_dt')

        date = pd.date_range(START, END, freq='D')
        sale_amt = []
        for d in date:
            df_temp = df_order[df_order['ord_dt'] == d]
            sale_amt.append(df_temp['sale_amt'].sum())

        df_res = pd.DataFrame({'sale_amt': sale_amt, 'ord_dt': date})
        df_res.to_csv(os.path.join(output_dir, 'shop_{0}.csv'.format(shop_id)), index=False)
        print("Finished generating salt_amt_by_day: {0}/{1}".format(shop_id, len(shop_id_list)))


def generate_ads_by_day(shop_id):
    """
    将商店在广告方面的数据 (charge/consume) 按天 (START, END) 进行整理
    TODO: 将每一次 consume 额度在该月内进行平均，或者在接下来一月内进行平均
    """
    date = pd.date_range(START, END, freq='D')
    df_ads_by_day = pd.DataFrame({'consume': [0.] * len(date),
                                  'charge': [0.] * len(date)}, 
                                  index=date)
    print(df_ads_by_day.head())

    df_ads = pd.read_csv(
            os.path.join(DATA_DIR, 'ads/shop_{0}.csv'.format(shop_id)), parse_dates=[0])
    df_ads.index = df_ads['create_dt'].tolist()

    for d in df_ads.index.tolist():
        consume = df_ads.loc[d]['consume']
        charge = df_ads.loc[d]['charge']
        df_ads_by_day.loc[d]['consume'] = consume
        df_ads_by_day.loc[d]['charge'] = charge

    output_dir = os.path.join(DATA_DIR, 'ads_by_day')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_ads_by_day.to_csv(
            os.path.join(output_dir, 'shop_{0}.csv'.format(shop_id)), 
            index=True, index_label='create_dt')


def get_sale_amt_seq(shop_id_list, seq_len):
    """
    sale_amt: [num_shops, num_days]
    valid_X: [num_shops, seq_len] (sale_amt[:, -90-seq_len:-90])
    valid_Y: [num_shops, 90] (sale_amt[:, -90:])
    train_X: [num_shops * (271 - 90 - seq_len -1), seq_len, 1]
    train_Y: [num_shops * (271 - 90 - seq_len -1), 1]
    """
    sale_amt_matrix = []
    for id in shop_id_list:
        filename = os.path.join(DATA_DIR, 'sale_amt_by_day', 'shop_{0}.csv'.format(id))
        df = pd.read_csv(filename)
        sale_amt = df['sale_amt'].tolist()
        sale_amt_matrix.append(sale_amt)

    sale_amt = np.asarray(sale_amt_matrix, dtype=np.float32)
    train_XY = sale_amt[:, :-90]
    valid_Y = sale_amt[:, -90:]
    valid_X = sale_amt[:, -90-seq_len:-90]
    valid_X = valid_X[:, :, np.newaxis]
    print('train_XY.shape: {0}'.format(train_XY.shape))
    print('valid_X.shape: {0}'.format(valid_X.shape))
    print('valid_Y.shape: {0}'.format(valid_Y.shape))

    train_X = []
    train_Y = []
    total_seq_len = train_XY.shape[1]
    for ii in range(total_seq_len - seq_len - 1):
        s, e = ii, ii + seq_len
        one_seq = train_XY[:, s:e]
        one_y = train_XY[:, e: e+1]
        train_X.append(one_seq)
        train_Y.append(one_y)

    train_X = np.concatenate(train_X, axis=0)
    train_Y = np.concatenate(train_Y, axis=0)
    train_X = train_X[:, :, np.newaxis]

    return {'train_X': train_X, 'train_Y': train_Y, 'valid_X': valid_X, 'valid_Y': valid_Y}
        

def build_model(seq_len):
    import tensorflow.contrib.keras as keras
    from tensorflow.contrib.keras import optimizers
    from tensorflow.contrib.keras import layers

    RNN = keras.layers.LSTM
    LAYERS = 3
    model = keras.models.Sequential()
    for _ in range(LAYERS):
        model.add(RNN(100, input_shape=(seq_len, 1), return_sequences=True))

    model.add(RNN(100, return_sequences=False))
    model.add(layers.Dense(1, activation='linear'))

    optimizer = optimizers.Adam(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print('model input shape: {0}'.format(model.input_shape))
    print('model output shape: {0}'.format(model.output_shape))
    return model


def evaluate(model, valid_X, valid_Y, seq_len):
    input_X = valid_X
    pred_Y = []
    for ii in range(90):
        pred_y = model.predict(input_X)
        pred_Y.append(pred_y)
        pred_y = pred_y[:, :, np.newaxis]
        input_X = np.concatenate([input_X[:, 1:], pred_y], axis=1)

    pred_Y = np.concatenate(pred_Y, axis=1)
    print('pred_Y.shape: {0}'.format(pred_Y.shape))
    print('valid_Y.shape: {0}'.format(valid_Y.shape))
    pred_Y = np.sum(pred_Y, axis=1)
    valid_Y = np.sum(valid_Y, axis=1)
    score = np.sum(np.abs(pred_Y - valid_Y)) / np.sum(valid_Y)
    print('score: {0}'.format(score))
    return score



def main():
    month_ends = ['2016-06-30', '2016-07-31', '2016-08-31', '2016-09-30', 
            '2016-10-31', '2016-11-30', '2016-12-31', '2017-01-31',
            '2017-02-28', '2017-03-31', '2017-04-30']

    num_shops = 1000

    #generate_csv_by_shop()
    #generate_ads_by_day(shop_id)
    #generate_sale_amt_by_day(range(1, 3001))
    seq_len = 10
    BATCH_SIZE = 32
    epochs = 40
    data = get_sale_amt_seq(range(1, num_shops), seq_len=seq_len)
  
    model = build_model(seq_len)
    scores = []
    for ii in range(epochs):
        score = evaluate(model, data['valid_X'], data['valid_Y'], seq_len)
        scores.append(score)
        model.fit(data['train_X'], data['train_Y'], batch_size=BATCH_SIZE, epochs=1)

    df_metrics = pd.DataFrame({'epochs': range(epochs), 'scores': scores})
    df_metrics.to_csv(os.path.join(DATA_DIR, 'logger.csv'))



    


if __name__ == '__main__':
    main()
