# 一个简单的神经网络模型
import numpy as np
import time

class NeuralNetwork():

    def __init__(self):
        # 设置随机数种子
        np.random.seed(int(time.time()))

        # 将权重转化为一个3*1的矩阵，其值分布为-1~1，并且均值为0
        self.synaptic_weights = 2 * np.random.random((3,1)) -1 
        self.synaptic_b = np.random.random(1)

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
        # 一般在做sigmod之前，还会加个值B
        output = self.sigmod(np.dot(inputs,self.synaptic_weights))
        return output

    # 使用吴恩达的神经网络和深度学习中的损失函数，和计算方法
    def train2(self,training_inputs,training_outputs,training_iterations):
        # 学习率
        i = 0.5
        # 训练模型
        for iteration in range(training_iterations):
            # 得到输出
            output = self.think2(training_inputs)
            
            # 梯度下降算法，计算反向传翻的调整值
            # 损失函数对输入与w和b的结果的求导，最终公式
            dz = output - training_outputs
            # 计算对w的导数，最终公式
            dw = 1/3 * np.dot(training_inputs.T,dz)
            ## 计算对b的导数，最终公式
            db = 1/3 * np.sum(dz)

            self.synaptic_weights -= i*dw
            self.synaptic_b -= i*db

    # 使用吴恩达的神经网络和深度学习中的损失函数，和计算方法
    def think2(self,inputs):
        inputs = inputs.astype(float)
        # 一般在做sigmod之前，还会加个值
        output = self.sigmod(np.dot(inputs,self.synaptic_weights) + self.synaptic_b)
        return output

if __name__ == "__main__":
    # 初始化神经类
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights1:")
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
    print("------------------------------------------")
    # 初始化神经类
    neural_network2 = NeuralNetwork()

    print("Beginning Randomly Generated Weights:")
    print(neural_network2.synaptic_weights)

    # 训练数据
    training_inputs = np.array([[0,0,1],
                               [1,1,1],
                               [1,0,1],
                               [0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T

    # 开始训练
    neural_network2.train2(training_inputs,training_outputs,15000)

    print("Ending Weights After Training: ")
    print(neural_network2.synaptic_weights)
    print(neural_network2.synaptic_b)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))

    print("Considering New Situation: ",user_input_one,user_input_two,user_input_three)
    print("New Output data: ")
    print(neural_network2.think2(np.array([user_input_one,user_input_two,user_input_three])))
    print("Wow,we did it!")