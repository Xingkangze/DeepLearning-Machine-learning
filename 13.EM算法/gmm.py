# --encoding:utf-8 --
"""
实现GMM高斯混合聚类
根据EM算法流程实现这个流程
http://scipy.github.io/devdocs/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal
"""
import numpy as np
from scipy.stats import multivariate_normal


def train(x, max_iter=100):
    """
    进行GMM模型训练，并返回对应的μ和σ的值(假定x数据中的簇类别数目为2)
    :param x: 输入的特征矩阵x
    :param max_iter:  最大的迭代次数
    :return:  返回一个五元组(pi, μ1， μ2，σ1，σ2)
    """
    # 1. 获取样本的数量m以及特征维度n
    m, n = np.shape(x)

    # 2. 初始化相关变量
    # 以每一列中的最小值作为mu1，mu1中的元素数目就是列的数目(n)个
    mu1 = x.min(axis=0)
    mu2 = x.max(axis=0)
    sigma1 = np.identity(n)
    sigma2 = np.identity(n)
    pi = 0.5

    # 3. 实现EM算法
    for i in range(max_iter):
        # a. 初始化多元高斯分布（初始化两个多元高斯混合概率密度函数）
        norm1 = multivariate_normal(mu1, sigma1)
        norm2 = multivariate_normal(mu2, sigma2)

        # E step
        # 计算所有样本数据在norm1和norm2中的概率
        tau1 = pi * norm1.pdf(x)
        tau2 = (1 - pi) * norm2.pdf(x)
        # 概率做一个归一化操作
        w = tau1 / (tau1 + tau2)

        # M step
        mu1 = np.dot(w, x) / np.sum(w)
        mu2 = np.dot(1 - w, x) / np.sum(1 - w)
        sigma1 = np.dot(w * (x - mu1).T, (x - mu1)) / np.sum(w)
        sigma2 = np.dot((1 - w) * (x - mu2).T, (x - mu2)) / np.sum(1 - w)
        pi = np.sum(w) / m

    # 返回最终解
    return (pi, mu1, mu2, sigma1, sigma2)


if __name__ == '__main__':
    np.random.seed(28)

    # 产生一个服从多元高斯分布的数据（标准正态分布的多元高斯数据）
    mean1 = (0, 0, 0)  # x1\x2\x3的数据分布都是服从正态分布的，同时均值均为0
    cov1 = np.diag((1, 1, 1))
    data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=500)

    # 产生一个数据分布不均衡
    mean2 = (2, 2, 3)
    cov2 = np.array([[1, 1, 3], [1, 2, 1], [0, 0, 1]])
    data2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=200)

    # 合并两个数据
    data = np.vstack((data1, data2))

    pi, mu1, mu2, sigma1, sigma2 = train(data, 100)
    print("第一个类别的相关参数:")
    print(mu1)
    print(sigma1)
    print("第二个类别的相关参数:")
    print(mu2)
    print(sigma2)

    print("预测样本属于那个类别(概率越大就是那个类别)：")
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    x = np.array([0, 1, 0])
    print(pi * norm1.pdf(x)) # 属于类别1的概率为:0.0275  => 0.989
    print((1 - pi) * norm2.pdf(x))# 属于类别1的概率为:0.0003 => 0.011
