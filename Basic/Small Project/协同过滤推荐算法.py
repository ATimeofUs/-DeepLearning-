from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
import seaborn as sns

def cost(params, Y, R, num_features, learning_rate):
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)

    # add the cost regularization
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X) # (1682, 10)
    Theta_grad = (error.T * X) + (learning_rate * Theta) # (943, 10)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad


def show_recommendations(Y_predict, R, user_id=0):
    """
    Y_predict: 训练好的预测矩阵 (num_movies, num_users)
    R: 原始评分标记矩阵
    user_id: 想要查看的用户索引
    """
    # 1. 提取该用户的预测评分（转为数组处理更方便）
    user_pred_scores = np.array(Y_predict[:, user_id]).flatten()

    # 2. 找出该用户未评分过的电影索引 (R == 0)
    unrated_indices = np.where(R[:, user_id] == 0)[0]

    # 3. 仅在未评分的电影中进行排序
    # 技巧：通过索引切片获取未评分电影的得分，排序，然后取倒序前10
    unrated_scores = user_pred_scores[unrated_indices]
    top_n_relative_indices = np.argsort(unrated_scores)[::-1][:10]

    # 4. 映射回原始电影列表的索引
    top_10_movie_indices = unrated_indices[top_n_relative_indices]

    print(f"\n--- 为用户 {user_id} 推荐的前10部电影 (预测分) ---")
    for i in top_10_movie_indices:
        print(f"预测评分 {user_pred_scores[i]:.2f} 星: {i}")


def plot_matrices_comparison(Y_original, Y_predict):
    # 取前10个用户和前10部电影
    original_sub = Y_original[:10, :10]
    predict_sub = Y_predict[:10, :10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 训练前（原始数据，含大量0）
    sns.heatmap(original_sub, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax1)
    ax1.set_title("Original Ratings (Top 10x10)")
    ax1.set_xlabel("User ID")
    ax1.set_ylabel("Movie ID")

    # 训练后（预测数据，被补全）
    sns.heatmap(predict_sub, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax2)
    ax2.set_title("Predicted Ratings (Top 10x10)")
    ax2.set_xlabel("User ID")
    ax2.set_ylabel("Movie ID")

    plt.show()

def main():
    data = loadmat(r'D:\__SelfCoding\AI_learning\fengdu78 Coursera-ML-AndrewNg-Notes master code\ex8-anomaly detection and recommendation\data\ex8_movies.mat')
    params_data = loadmat(r'D:\__SelfCoding\AI_learning\fengdu78 Coursera-ML-AndrewNg-Notes master code\ex8-anomaly detection and recommendation\data\ex8_movieParams.mat')

    # 电影数量1682，用户数量943
    # Y是电影评分矩阵，R是二进制“是否”评分矩阵
    Y = data['Y'] # (1682, 943)
    R = data['R'] # (1682, 943)
    learning_rate = 0.4

    # 电影特征矩阵X，用户偏好矩阵Theta
    X = params_data['X'] # (1682, 10)
    Theta = params_data['Theta'] # (943, 10)

    num_features = X.shape[1]

    # 均值归一化矩阵
    def normalize_ratings(Y, R):
        m, n = Y.shape
        Ymean = np.zeros((m, 1))
        Ynorm = np.zeros(Y.shape)
        for i in range(m):
            idx = np.where(R[i, :] == 1)[0]
            if len(idx) > 0:
                Ymean[i] = Y[i, idx].mean()
                Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        return Ynorm, Ymean

    Ynorm, Ymean = normalize_ratings(Y, R)
    params = np.concatenate((X.ravel(), Theta.ravel()))

    res = minimize(
        fun=cost,
        x0=params,
        args=(Ynorm, R, num_features, learning_rate),
        method='CG',
        jac=True,
        options={'maxiter': 100}
    )

    params = res.x
    X_new, Theta_new = np.matrix(np.reshape(params[:Y.shape[0] * num_features], (Y.shape[0], num_features))), \
                       np.matrix(np.reshape(params[Y.shape[0] * num_features:], (Y.shape[1], num_features)))
    Y_predict = X_new @ Theta_new.T + Ymean

    print("Training completed.")

    plot_matrices_comparison(Y, Y_predict)


if __name__ == '__main__':
    main()