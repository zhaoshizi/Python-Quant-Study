import numpy as np


def sn_random_numbers(shape, antithetic=True, moment_matching=True, fixed_seed=False):
    """
    Returns an array of shape with (pseduo)random numbers
    that are standard normally distributed

    parameters
    ------------
    shape: tuple(o,n,m)
        generation of array with shape(o,n,m)
    antithetic: Boolean
        generation of antithetic variates对偶
    moment_matching:Boolean 矩匹配
        matching of first and second moments

    Results
    -------------
    ran: (o,n,m) array of (pseudo)random numbers
    """
    if fixed_seed:
        np.random.seed(1000)
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2]/2))
        # 拼接在一起
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        ran = np.random.standard_normal(shape)
    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    if shape[0] == 1:
        return ran[0]
    else:
        return ran

if __name__ == '__main__':
    snrn = sn_random_numbers((2,2,2),antithetic = False,moment_matching= False,fixed_seed=True)
    print(snrn)