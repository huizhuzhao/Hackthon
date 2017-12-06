# encoding: utf-8
# author: huizhu
# created time: 2017年12月06日 星期三 19时02分26秒

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '/home/xtalpi/datasets/test_data/jddjr/salesForecast'
NAME = ['comment', 'order', 'ads', 'product', 'sales_sum']
DATE = {'comment': 'create_dt', 
                'order': 'ord_dt', 
                'ads': 'create_dt', 
                'product': 'on_dt', 
                'sales_sum': 'dt'}

def generate_csv_by_shop():
    for n in NAME:
        df = pd.read_csv(os.path.join(data_dir, 't_{0}.csv'.format(n)), parse_dates=[0])
        sub_dir = os.path.join(data_dir, n)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        for ii in range(1, 3001):
            df_sub = df[df.shop_id == ii]
            df_sub = df_sub.sort_values(by=DATE[n])
            filename = os.path.join(sub_dir, 'shop_{0}.csv'.format(ii))
            df_sub.to_csv(filename, index=False)


def generate_sale_amt_by_day(shop_id):
    output_dir = os.path.join(data_dir, 'sale_amt_by_day')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    order_file = os.path.join(data_dir, 'order/shop_{0}.csv'.format(shop_id))
    df_order = pd.read_csv(order_file, parse_dates=[0])
    df_order.index = df_order.ord_dt
    df_order = df_order.sort_values(by='ord_dt')

    date = pd.date_range('2016-08-03', '2017-04-30', freq='D')

    sale_amt = []
    for d in date:
        df_temp = df_order[df_order['ord_dt'] == d]
        sale_amt.append(df_temp['sale_amt'].sum())

    df_res = pd.DataFrame({'sale_amt': sale_amt, 'ord_dt': date})
    df_res.to_csv(os.path.join(output_dir, 'shop_{0}.csv'.format(shop_id)), index=False)


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

    date_range = pd.date_range('2016-08', '2017-04', freq='M')
    print(date_range[:10])
    df_ads = pd.read_csv(os.path.join(data_dir, 'ads/shop_{0}.csv'.format(shop_id)), parse_dates=[0], date_parser=dateparse)
    df_ads.index = df_ads['create_dt']
    print(df_ads.head())
    print(df_ads['2016-09'])



    


if __name__ == '__main__':
    main()
