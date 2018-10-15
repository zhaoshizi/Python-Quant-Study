# 一个简单的神经网络模型
import numpy as np

class NeuralNetwork():

    def __init__(self):
        # 设置随机数种子
        np.random.seed(1)

        # 将权重转化为一个3*1的矩阵，其值分布为-1~1，并且均值为0
        self.synaptic_weights = 2 * np.random.random((3,1)) -1 

    def sigmod(self,x):
        # 应用sigmod激活函数
        return 1 / (1 + np.exp(-x))

    def sigmod_derivative(self,x):
        # 计算Sigmod函数的偏导数
        return x * (1-x)

    def train(self,training_inputs,training_outputs,training_iterations):

        # 训练模型
        for iteration in range(training_iterations):
            # 得到输出
            output = self.think(training_inputs)

            # 计算误差（会有更复杂的计算误差的方法）
            error = training_outputs - output

            # 微调权重
            adjustments = np.dot(training_inputs.T,error*self.sigmod_derivative(output))

            self.synaptic_weights += adjustments

    def think(self,inputs):
        # 输入通过网络得出输出
        # 转化为浮点型数据类型

        inputs = inputs.astype(float)
        # 一般在做sigmod之前，还会加个值
        output = self.sigmod(np.dot(inputs,self.synaptic_weights))
        return output

if __name__ == "__main__":
    # 初始化神经类
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights:")
    print(neural_network.synaptic_weights)

    # 训练数据
    training_inputs = np.array([[0,0,1],
                               [1,1,1],
                               [1,0,1],
                               [0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T

    # 开始训练
    neural_network.train(training_inputs,training_outputs,15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))

    print("Considering New Situation: ",user_input_one,user_input_two,user_input_three)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one,user_input_two,user_input_three])))
    print("Wow,we did it!")