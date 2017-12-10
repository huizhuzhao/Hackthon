# encoding: utf-8
# author: huizhu
# created time: 2017年12月10日 星期日 12时40分19秒

from sklearn.preprocessing import StandardScaler
from Hackthon.jddjr import utils



def main():
    sale_amt = utils.get_sale_amt_by_day(range(1, 100))
    print(sale_amt.shape)

    scaler = StandardScaler()
    scaler.fit(sale_amt.T)
    sale_amt_t = scaler.transform(sale_amt.T).T

    print(sale_amt_t.shape)


if __name__ == '__main__':
    main()
