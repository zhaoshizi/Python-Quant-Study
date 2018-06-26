# 伯努利分布测试
# 在Numpy中使用numpy.random.binomial(1,p)来获取1的概率为p的前提下，生成的随机变量
# 实现函数casino()：假设有100个赌徒，每个赌徒都有1000000元，并且每个都想在赌场玩1000万次，
# 在不同的胜率、赔率和手续费下casino()函数返回总体统计结果

import numpy as np 
import matplotlib.pyplot as plt
# 设置100个赌徒
gamblers = 100


def casino(win_rate, win_once=1, loss_once=1, commission=0.01):
    """
        赌场：简单设定每个赌徒都有1000000元，并且每个赌徒都想在赌场玩10000000次，但是如果没钱了就别想玩了
        win_rate:输赢的概率
        win_once:每次赢的钱数
        loss_once:每次输的钱数
        commission:手续费这里简单设置为0.01，即1%
    """
    my_money = 100
    play_cnt = 1000
    commission = commission
    for _ in np.arange(0, play_cnt):
        # 使用伯努利分布，根据win_rate来获取输赢
        w = np.random.binomial(1, win_rate)
        if w:
            # 赢了 +win_once
            my_money += win_once
        else:
            # 输了 -loss_once
            my_money -=loss_once
        # 手续费
        my_money -= commission
        if my_money <= 0:
            # 没钱就别玩了，不赊账
            break
    return my_money

# 100个赌徒进场天堂赌场，胜率0.5，赔率1，还没手续（没老千，没抽头）
heaven_moneys = [casino(0.5, commission=0) for _ in np.arange(0,gamblers)]

# 100个赌徒进有老千的赌场，胜率0.4， 赔率1，无手续费
cheat_moneys = [casino(0.4,commission=0) for _ in np.arange(0,gamblers)]

# 100个赌徒进有手续费的赌场，胜率0.5，赔率1，手续费0.01
commison_moneys = [casino(0.5,commission=0.01) for _ in np.arange(0,gamblers)]

# 画出天堂赌场赌徒结果的直方图，时间比较长
_ = plt.hist(heaven_moneys,bins=30)
plt.show()
# 画出有老千赌场结果的直方图
#_ = plt.hist(cheat_moneys,bins= 30)

# 画出有手续费赌场结果的直方图
#_ = plt.hist(commison_moneys,bins = 30)

# 如果提高赔率，胜率0.5，赔率1.04，手续费0.01
moneys = [casino(0.5,commission = 0.01,win_once=1.02,loss_once = 0.98) for _ in np.arange(0,gamblers)]
_ = plt.hist(moneys,bins =30)
# 如果降低胜率，胜率0.45，赔率1.04，手续费0.01
moneys = [casino(0.45,commission=0.01,win_once=1.02,loss_once=0.98) for _ in np.arange(0,gamblers)]
_ = plt.hist(moneys,bins =30)

