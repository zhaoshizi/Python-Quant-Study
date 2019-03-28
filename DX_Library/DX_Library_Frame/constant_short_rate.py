# 固定短期利率下的风险中立折现类
# 
# DX Library Frame
# constant_short_rate.py
#
from get_year_deltas import *

class constant_short_rate(object):
    """
    Class for constant short rate discounting.

    Attributes
    ----------
    name : string
    short_rate : float (positive)
        constant rate for discounting

    Methods
    ---------
    get_discount_factors:
        get discount factors given a list/array of datetime objects or year fractions
    """
    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0 :
            raise ValueError('Short rate negative.')

    def get_diccount_factors(self, date_list, dtobjects=True):
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        # 折现的计算
        dflist = np.exp(self.short_rate * np.sort(-1 * dlist))
        return np.array((date_list, dflist)).T

if __name__ == '__main__':
    import datetime as dt
    dates = [dt.datetime(2015,1,1), dt.datetime(2015,7,1), dt.datetime(2016,1,1)]
    csr = constant_short_rate('csr',0.05)

    f1 = csr.get_diccount_factors(dates)
    print(f1)

    deltas = get_year_deltas(dates)
    f2 = csr.get_diccount_factors(deltas, dtobjects=False)
    print(f2)
