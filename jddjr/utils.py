# encoding: utf-8
# author: huizhu
# created time: 2017年12月06日 星期三 19时02分26秒

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.keras import callbacks


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


def generate_features_by_day(shop_id_list, dir_name='features'):
    """
    因为商店每天的销售额数据会有多条, 本函数会将某个商店 (shop_id) 的销售额数据 (sale_amt) 
    依据 天 (2016-08-03 至 2017-04-03) 进行求和，将结果写入 sale_amt_by_day/shop_i.csv 中
    """
    output_dir = os.path.join(DATA_DIR, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_day_feature(date):
        df = pd.DataFrame({'ord_dt': date})
        df['ord_dt'] = pd.to_datetime(df['ord_dt'])
        feat = df['ord_dt'].dt.dayofweek
        feat = np.eye(7)[feat].astype(np.int32)
        d = {}
        for ii in range(7):
            d['day_{0}'.format(ii)] = feat[:, ii].tolist()

        return d

    def sort_df(filename, name):
        dt_col = DATE[name]
        df = pd.read_csv(filename, parse_dates=[0])
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(by=dt_col)

        return df


    for shop_id in shop_id_list:
        order_file = os.path.join(DATA_DIR, 'order/shop_{0}.csv'.format(shop_id))
        ads_file = os.path.join(DATA_DIR, 'ads/shop_{0}.csv'.format(shop_id))

        df_order = sort_df(order_file, 'order')
        df_ads = sort_df(ads_file, 'ads')

        date = pd.date_range(START, END, freq='D')
        sale_amt = []
        ads_charge = np.zeros((len(date), ), dtype=np.float32)
        ads_consume = np.zeros((len(date), ), dtype=np.float32)
        ads_lag = 30
        for ii, d in enumerate(date):
            df_order_tmp = df_order[df_order['ord_dt'] == d]
            df_ads_tmp = df_ads[df_ads['create_dt'] == d]

            sale_amt.append(df_order_tmp['sale_amt'].sum())
            ads_charge[ii:ii+ads_lag] += df_ads_tmp['charge'].sum()
            ads_consume[ii:ii+ads_lag] += df_ads_tmp['consume'].sum()
            #ads_charge.append(df_ads_tmp['charge'].sum())
            #ads_consume.append(df_ads_tmp['consume'].sum())

        day_feat = get_day_feature(date)
        features = day_feat
        features.update({'sale_amt': sale_amt,
                         'ord_dt': date,
                         'ads_charge': ads_charge,
                         'ads_consume': ads_consume})
        df_res = pd.DataFrame(features)

        output_file = os.path.join(output_dir, 'shop_{0}.csv'.format(shop_id))
        df_res.to_csv(output_file, index=False)
        print("Finished generating {0}: {1}/{2}".format(output_file, shop_id, len(shop_id_list)))


def get_features(shop_id_list, dir_name='features'):
    sale_amt_matrix = []
    features = []
    cols = ['sale_amt', 'ads_consume', 
            'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6']

    for id in shop_id_list:
        filename = os.path.join(DATA_DIR, dir_name, 'shop_{0}.csv'.format(id))
        df = pd.read_csv(filename)
        df['ord_dt'] = pd.to_datetime(df['ord_dt'])
        #df.index = df['ord_dt'].tolist()
        #df = df.drop(['2016-11-11'])
        feat = df[cols].values
        features.append(feat)

    features = np.asarray(features, dtype=np.float32)
    print('features.shape: {0}'.format(features.shape))
    return features

    

def get_train_data(sale_amt, seq_len, features):
    train_XY = sale_amt[:, :-90]
    valid_Y = sale_amt[:, -90:]
    valid_X = train_XY[:, -seq_len:]
    valid_X = valid_X[:, :, np.newaxis]
    print('train_XY.shape/max: {0}/{1}'.format(train_XY.shape, np.max(train_XY)))
    print('valid_X.shape/max: {0}/{1}'.format(valid_X.shape, np.max(valid_X)))
    print('valid_Y.shape/max: {0}/{1}'.format(valid_Y.shape, np.max(valid_Y)))

    train_X = []
    train_Y = []
    total_seq_len = train_XY.shape[1]
    slides = total_seq_len - seq_len - 1
    print('slides: {0}'.format(slides))
    for ii in range(slides):
        s, e = ii, ii + seq_len
        one_seq = train_XY[:, s:e]
        one_y = train_XY[:, e: e+1]
        train_X.append(one_seq)
        train_Y.append(one_y)

    train_X = np.concatenate(train_X, axis=0)
    train_Y = np.concatenate(train_Y, axis=0)
    train_X = train_X[:, :, np.newaxis]

    print('train_X.shape/max: {0}/{1}'.format(train_X.shape, np.max(train_X)))
    print('train_Y.shape/max: {0}/{1}'.format(train_Y.shape, np.max(train_Y)))

    class Datat(object):
        pass

    datat = Datat()
    datat.train_X = train_X
    datat.train_Y = train_Y
    datat.valid_X = valid_X
    datat.valid_Y = valid_Y
    datat.scaler = None

    return datat

def _train_X_train_Y(train_XY, input_seq_len, output_seq_len):
    train_X = []
    train_Y = []
    total_seq_len = train_XY.shape[1]
    slides = total_seq_len - input_seq_len - output_seq_len + 1
    assert slides > 0, ("slides: {0}".format(slides))
    print('slides: {0}'.format(slides))
    for ii in range(slides):
        e1, e2 = ii, ii + input_seq_len
        one_seq = train_XY[:, e1:e2]
        one_y = train_XY[:, e2:e2+output_seq_len, 0:1]
        train_X.append(one_seq)
        train_Y.append(one_y)

    train_X = np.concatenate(train_X, axis=0)
    train_Y = np.concatenate(train_Y, axis=0)
    print('train_X.shape/max: {0}/{1}'.format(train_X.shape, np.max(train_X)))
    print('train_Y.shape/max: {0}/{1}'.format(train_Y.shape, np.max(train_Y)))
    return train_X, train_Y


def get_seq2seq_data(features, input_seq_len, output_seq_len):
    train_XY = features[:, :-output_seq_len]
    valid_X = train_XY[:, -input_seq_len:]
    valid_Y = features[:, -output_seq_len:, 0:1]

    print('train_XY.shape/max: {0}/{1}'.format(train_XY.shape, np.max(train_XY)))
    print('valid_X.shape/max: {0}/{1}'.format(valid_X.shape, np.max(valid_X)))
    print('valid_Y.shape/max: {0}/{1}'.format(valid_Y.shape, np.max(valid_Y)))

    train_X, train_Y = _train_X_train_Y(train_XY, input_seq_len, output_seq_len)
    class Datat(object):
        pass

    datat = Datat()
    datat.train_X = train_X
    datat.train_Y = train_Y
    datat.valid_X = valid_X
    datat.valid_Y = valid_Y

    return datat

def resample_date(start_date, end_date, gap, shop_id=None): 
    """
    针对order数据，提供开始日期，结束日期和采样周期，将自动在时间区间采样，
    采样周期内的sale值做sum处理变成一个点
    """
    start_sec = time.mktime(time.strptime(start_date,'%Y-%m-%d'))
    end_sec = time.mktime(time.strptime(end_date,'%Y-%m-%d'))
    period = int((end_sec - start_sec)/(24*60*60))+1

    # sift order models
    if shop_id is not None:
        order_sift = order[(order['ord_dt']>=start_date) & (order.ord_dt<=end_date)&(order.shop_id==shop_id)].loc[:, ['ord_dt', 'sale_amt', 'rtn_amt', 'shop_id']]
    else:
        order_sift = order[(order['ord_dt']>=start_date) & (order.ord_dt<=end_date)].loc[:, ['ord_dt', 'sale_amt', 'rtn_amt', 'shop_id']]
    order_sift = order_sift.groupby(['ord_dt']).sum()
    # sales - return money
    sales = order_sift.sale_amt - order_sift.rtn_amt
    sales_amt = pd.DataFrame({'sales': sales.values}, index=order_sift.index.values)
    date_range = pd.date_range(start_date, periods=period, feq='D')
    date_range = date_range.strftime('%Y-%m-%d')
    time_df = pd.DataFrame(index=date_range )

    # combine two dataframes and get the sum sale dataframe
    merge_df = pd.concat([sales_amt, time_df], axis=1).fillna(0)
    merge_df['group_index'] = gen_group_index(period, gap)
    merge_df = merge_df.groupby(['group_index']).sum()

    # generate new date index
    freq = '%sD' % gap
    period_index = pd.date_range(start_date, end_date, freq=freq)
    merge_df['period_index'] = period_index
    return merge_df

# 定义产生分组索引的函数，比如我们要计算的周期是 20 天，则按照日期，20 个交易日一组
def gen_group_index(total, group_len):
    """ generate an item group index array

    suppose total = 10, unitlen = 2, then we will return array [0 0 1 1 2 2 3 3 4 4]
    """

    group_count = total / group_len
    group_index = np.arange(total)
    for i in range(group_count):
        group_index[i * group_len: (i + 1) * group_len] = i
    group_index[(i + 1) * group_len : total] = i + 1
    return group_index.tolist()


def get_callbacks(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    weights_filename = os.path.join(log_dir, 'weights_{epoch:02d}_{loss:.2f}.hdf5')
    ckpt = callbacks.ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True)

    logger_filename = os.path.join(log_dir, 'logger.csv')
    csv_logger = callbacks.CSVLogger(logger_filename, append=True)

    return [ckpt, csv_logger]



