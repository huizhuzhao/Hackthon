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

def get_sale_amt_by_day(shop_id_list):
    sale_amt_matrix = []
    for id in shop_id_list:
        filename = os.path.join(DATA_DIR, 'sale_amt_by_day', 'shop_{0}.csv'.format(id))
        df = pd.read_csv(filename)
        df.index = df['ord_dt'].tolist()
        #df = df.drop(['2016-11-11'])
        sale_amt = df['sale_amt'].tolist()
        sale_amt_matrix.append(sale_amt)

    sale_amt = np.asarray(sale_amt_matrix, dtype=np.float32)

    return sale_amt


class DataTransform(object):
    def __init__(self, sale_amt, seq_len):
        self.sale_amt = sale_amt
        self.seq_len = seq_len
        self.train_XY = sale_amt[:, :-90]
        self.valid_Y = sale_amt[:, -90:]
        valid_X = self.train_XY[:, -self.seq_len:]
        self.valid_X = valid_X[:, :, np.newaxis]
        print('self.train_XY.shape/max: {0}/{1}'.format(self.train_XY.shape, np.max(self.train_XY)))
        print('self.valid_X.shape/max: {0}/{1}'.format(self.valid_X.shape, np.max(self.valid_X)))
        print('self.valid_Y.shape/max: {0}/{1}'.format(self.valid_Y.shape, np.max(self.valid_Y)))

    def scale(self):
        train_XY = self.train_XY
        self.scaler = StandardScaler()
        self.scaler.fit(train_XY.T)
        train_XY = self.scaler.transform(train_XY.T).T
        #valid_Y = self.scaler.transform(valid_Y.T).T
        valid_X = train_XY[:, -self.seq_len:]
        valid_X = valid_X[:, :, np.newaxis]

        self.train_XY = train_XY
        self.valid_X = valid_X
        print('self.train_XY.shape/max: {0}/{1}'.format(train_XY.shape, np.max(train_XY)))
        print('self.valid_X.shape/max: {0}/{1}'.format(valid_X.shape, np.max(valid_X)))


    def get_train_data(self):
        train_XY = self.train_XY
        seq_len = self.seq_len
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
        self.train_X = train_X
        self.train_Y = train_Y

        print('self.train_X.shape/max: {0}/{1}'.format(train_X.shape, np.max(train_X)))
        print('self.train_Y.shape/max: {0}/{1}'.format(train_Y.shape, np.max(train_Y)))



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
