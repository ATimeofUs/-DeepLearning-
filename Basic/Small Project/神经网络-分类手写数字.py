import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import time

# 保持常量不变
INPUT_SIZE = 400
HIDDEN_SIZE = 25
NUM_LABELS = 10
LEARNING_RATE = 1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)


def nn_cost_and_gradient(params, X, y, input_size, hidden_size, num_labels, reg_lambda):
    """
    核心修改：一个函数完成前向和反向传播，全部使用向量化计算
    """


    m = X.shape[0]
    # 重构权重 (使用 np.array)
    theta1 = params[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)
    theta2 = params[hidden_size * (input_size + 1):].reshape(num_labels, hidden_size + 1)

    # --- 1. 前向传播 ---
    a1 = np.insert(X, 0, 1, axis=1)  # 添加偏置项 (5000, 401)
    z2 = a1 @ theta1.T  # (5000, 25)
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)  # 添加偏置项 (5000, 26)
    z3 = a2 @ theta2.T  # (5000, 10)
    h = sigmoid(z3)  # 预测输出 (5000, 10)

    # --- 2. 计算 Cost (J) ---
    # 使用 * 进行按元素相乘，np.sum 进行求和
    data_loss = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m
    reg_loss = (reg_lambda / (2 * m)) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))
    J = data_loss + reg_loss

    # --- 3. 反向传播 (梯度计算) ---
    d3 = h - y  # 输出层误差 (5000, 10)
    # 隐藏层误差：注意去掉 theta2 第一列(bias)，并使用 * 乘以上一层的梯度
    d2 = (d3 @ theta2[:, 1:]) * sigmoid_gradient(z2)  # (5000, 25)

    grad1 = (d2.T @ a1) / m
    grad2 = (d3.T @ a2) / m

    # --- 4. 梯度正则化 ---
    # 偏置列（第一列）不进行正则化
    grad1[:, 1:] += (reg_lambda / m) * theta1[:, 1:]
    grad2[:, 1:] += (reg_lambda / m) * theta2[:, 1:]

    # 将梯度展平为一维数组
    grad = np.concatenate([grad1.ravel(), grad2.ravel()])

    return J, grad


def predict(theta1, theta2, X):
    """
    给定训练好的权重，预测样本的标签
    """
    m = X.shape[0]

    # --- 前向传播 ---
    # 输入层 -> 隐藏层
    a1 = np.insert(X, 0, 1, axis=1)  # 添加偏置项 (m, 401)
    z2 = a1 @ theta1.T  # (m, 25)
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)  # (m, 26)

    # 隐藏层 -> 输出层
    z3 = a2 @ theta2.T  # (m, 10)
    h = sigmoid(z3)  # (m, 10)

    # --- 获取预测值 ---
    # np.argmax 返回每行最大值的索引（0-9）
    # 针对吴恩达的数据集，数字 0 被标记为 10，所以索引需要 +1
    # 这样索引 0 变成标签 1，索引 9 变成标签 10
    p = np.argmax(h, axis=1) + 1

    return p

def main():
    start_time = time.perf_counter()

    data = loadmat(
        r'D:\__SelfCoding\AI_learning\fengdu78 Coursera-ML-AndrewNg-Notes master code\ex4-NN back propagation\ex4data1.mat'
    )

    X = data['X']
    y = data['y']

    # One-Hot 编码
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # 随机初始化参数 (避免对称性断裂)
    epsilon = 0.12
    initial_params = np.random.uniform(
        -epsilon, epsilon,
        HIDDEN_SIZE * (INPUT_SIZE + 1) + NUM_LABELS * (HIDDEN_SIZE + 1))

    # 使用 TNC 算法，并传入 jac=True
    res = minimize(fun=nn_cost_and_gradient,
                   x0=initial_params,
                   args=(X, y_onehot, INPUT_SIZE, HIDDEN_SIZE, NUM_LABELS, LEARNING_RATE),
                   method='TNC',
                   jac=True,
                   options={'maxiter': 250})


    # 2. 从优化结果 res.x 中提取训练好的权重
    final_theta = res.x
    theta1 = final_theta[:HIDDEN_SIZE * (INPUT_SIZE + 1)].reshape(HIDDEN_SIZE, INPUT_SIZE + 1)
    theta2 = final_theta[HIDDEN_SIZE * (INPUT_SIZE + 1):].reshape(NUM_LABELS, HIDDEN_SIZE + 1)

    # 3. 进行预测
    predictions = predict(theta1, theta2, X)
    y_true = y.flatten()
    accuracy = np.mean(predictions == y_true) * 100

    end_time = time.perf_counter()

    return f"训练集准确率: {accuracy:.2f}%，用时: {end_time - start_time:.2f} 秒"


def main2():
    start_time = time.perf_counter()

    # 1. 加载数据
    data = loadmat(
        r'D:\__SelfCoding\AI_learning\fengdu78 Coursera-ML-AndrewNg-Notes master code\ex4-NN back propagation\ex4data1.mat')
    X = data['X'].astype('float32')  # 转换为 float32 提高 TF 计算效率
    y = data['y']

    # 2. 标签预处理 (One-Hot 编码)
    # y 最初是 5000-10，OneHotEncoder 会将其转换为 10 维向量
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # 3. 构建模型
    model = models.Sequential([
        # 修正点：input_shape 必须是 (400,)，代表 400 个特征的一维向量
        # 不要写成 (X.shape[0], ...) 也不要写成 (400, 1)
        layers.Input(shape=(400,)),
        layers.Dense(175, activation='relu'), # relu函数是max(0,x)
        layers.Dense(10, activation='softmax')  # 输出层用 softmax 更适合多分类
    ])

    # 4. 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # 配合 one-hot 编码的损失函数
        metrics=['accuracy']
    )

    # 5. 训练模型
    # verbose=1 可以让你看到每个 Epoch 的训练进度
    model.fit(X, y_onehot, epochs=50, batch_size=128, verbose=0)

    # 6. 获取预测值
    # 直接使用内置 evaluate 也能看准确率，这里保留你的逻辑
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(y_pred, axis=1)  # 得到 0-9

    # 注意：OneHotEncoder 对 1-10 进行编码时，
    # y_true_labels 得到的也是 0-9（对应原标签 1-10 的位置）
    y_true_labels = np.argmax(y_onehot, axis=1)

    # 7. 计算准确率
    accuracy = accuracy_score(y_true_labels, y_pred_labels)

    end_time = time.perf_counter()
    return f"训练集准确率为: {accuracy * 100:.2f}%，用时: {end_time - start_time:.2f} 秒"

if __name__ == '__main__':
    # str1 = main()

    str2 = main2()

    # print(str1)
    print(str2)

"""
模型1：训练集准确率: 97.94%，用时: 3.41 秒

模型2：训练集准确率为: 94.32%，用时: 3.39 秒

模型2 改进，多加一层：训练集准确率为: 96.06%，用时: 3.75 秒

relu 再改进训练集准确率为: 训练集准确率为: 99.94%，用时: 3.84 秒

多加很多神经元后：训练集准确率为: 100.00%，用时: 4.77 秒

"""