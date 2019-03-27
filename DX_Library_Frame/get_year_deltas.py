# 从datetime对象列表或者数组得年分数的函数
#
# DX Library Frame
# get_year_deltas.py
# 
import numpy as np 

def get_year_deltas(date_list, day_count=365.):
    """
    Return vector of floats with day deltas in years.
    Initial value normalized to zero.

    Parameters
    ----------
    date_list : list or array
        collection of datetime objects
    day_count : float
        number of days for a year
        (to account fot different conventions)

    Results
    ---------
    delta_list : array
        year fractions
    """

    start = date_list[0]
    delta_list = [(date - start).days / day_count for date in date_list]
    return np.array(delta_list)

if __name__ == "__main__":
    import datetime as dt
    dates = [dt.datetime(2015,1,1), dt.datetime(2015,7,1), dt.datetime(2016,1,1)]
    print(get_year_deltas(dates))


