from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from Basic.SoftMax回归 import accuracy


def load_data():
    # 直接加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 划分训练集和测试集，15%测试集，随机种子15179416016
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=123123
    )
    return X_train, X_test, y_train, y_test


def show_data(x, pos, y, feature_names=None):
    """
    展示第 pos+1 个样本的特征和标签

    参数:
        x: 特征矩阵 (numpy array 或 pandas DataFrame)
        pos: 样本位置索引 (int)
        y: 标签数组
        feature_names: 特征名称列表 (可选)
    """
    # 如果是 numpy array，转成 DataFrame
    if not isinstance(x, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(x.shape[1])]
        df = pd.DataFrame(x, columns=feature_names)
    else:
        df = x.copy()
        if feature_names is None:
            feature_names = df.columns

    # 取第 pos 样本
    sample = df.iloc[pos]
    label = y[pos]

    # 构建展示表格
    sample_df = pd.DataFrame(sample).T
    sample_df['target'] = label

    # 绘制表格
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=sample_df.values,
                     colLabels=sample_df.columns,
                     loc='center',
                     ax=None,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.show()

def main():
    X_train, X_test, y_train, y_test = load_data()

    # DMatrix要求将label直接封装进去
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 2. 将算法参数放入字典
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'num_parallel_tree': 100,

        'max_depth': 10,  # 限制树高，防止学得太深
        'min_child_weight': 5,  # 每个节点必须包含更多样本
        'subsample': 0.6,  # 更激进的行采样
        'colsample_bynode': 1,  # 保守的特征采样
        'lambda': 0.4,  # L2 正则化
        'alpha': 0.1,  # L1 正则化

        'learning_rate': 1,
    }

    # 3. 调用 train，注意 evals 的格式
    random_forest = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1,  # 随机森林只需要 1 轮迭代
        evals=[(dtrain, 'train'), (dtest, 'eval')]
    )

    print("Train is over")
    y_pred = random_forest.predict(dtest)

    sst = 0.5
    y_pred = [1 if y > sst else 0 for y in y_pred]
    ac = accuracy_score(y_test, y_pred)
    print("测试集准确率为: {:.2f}%".format(ac * 100))

if __name__ == '__main__':
    main()
