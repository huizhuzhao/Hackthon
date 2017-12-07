# encoding: utf-8
# author: huizhu
# created time: 2017年12月06日 星期三 19时02分26秒

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = '/home/xtalpi/datasets/test_data/jddjr/salesForecast'
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


def generate_sale_amt_by_day(shop_id):
    """
    因为商店每天的销售额数据会有多条, 本函数会将某个商店 (shop_id) 的销售额数据 (sale_amt) 
    依据 天 (2016-08-03 至 2017-04-03) 进行求和，将结果写入 sale_amt_by_day/shop_i.csv 中
    """
    output_dir = os.path.join(DATE_DIR, 'sale_amt_by_day')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    order_file = os.path.join(DATA_DIR, 'order/shop_{0}.csv'.format(shop_id))
    df_order = pd.read_csv(order_file, parse_dates=[0])
    df_order.index = df_order['ord_dt'].tolist()
    df_order = df_order.sort_values(by='ord_dt')

    date = pd.date_range('2016-08-03', '2017-04-30', freq='D')

    sale_amt = []
    for d in date:
        df_temp = df_order[df_order['ord_dt'] == d]
        sale_amt.append(df_temp['sale_amt'].sum())

    df_res = pd.DataFrame({'sale_amt': sale_amt, 'ord_dt': date})
    df_res.to_csv(os.path.join(output_dir, 'shop_{0}.csv'.format(shop_id)), index=False)


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



def main():
    shop_id = '1'
    month_ends = ['2016-06-30', '2016-07-31', '2016-08-31', '2016-09-30', 
            '2016-10-31', '2016-11-30', '2016-12-31', '2017-01-31',
            '2017-02-28', '2017-03-31', '2017-04-30']


    #generate_csv_by_shop()
    """
    df_comment = pd.read_csv(os.path.join(data_dir, 'comment/shop_{0}.csv'.format(shop_id)), parse_dates=[0])
    df_comment = df_comment.sort_values(by='create_dt')
    print(df_comment.head())
    """
    generate_ads_by_day(shop_id)




    


if __name__ == '__main__':
    main()
