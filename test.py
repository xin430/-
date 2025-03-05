import torch
import numpy as np
from sklearn.metrics import accuracy_score
from model import LSTM_Model  # 导入你的LSTM模型结构
from preprocess import load_and_preprocess_data

# 假设模型的相关参数
input_size = 78  # 特征数量
hidden_size = 128
num_classes = 8  # 类别数量（根据你的任务修改）
model = LSTM_Model(input_size, hidden_size, num_classes)

# 加载训练好的模型权重
model.load_state_dict(torch.load('lstm_model.pth'))  # 加载训练时保存的模型权重
model.eval()  # 切换到评估模式（推理模式）


# 进行预测并计算准确率
def predict(file_path):
    # 直接使用预处理函数来加载和处理数据
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_and_preprocess_data(file_path)

    # 使用测试集进行预测
    with torch.no_grad():  # 禁用梯度计算，节省内存
        outputs = model(X_test_tensor)  # 将测试数据传入模型

    # 使用 torch.max 获得每个样本的预测类别
    _, predicted = torch.max(outputs, 1)  # 预测类别是最大分数的索引

    # 计算准确率
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())  # 计算准确率
    print(f"Test Accuracy: {accuracy:.4f}")

    # 输出预测标签
    print("Predicted labels:", predicted.tolist())  # 将Tensor转换为列表输出


# CSV文件用于测试
file_path = 'data/CICIDS_test_data.csv'  # 替换为你的CSV文件路径

# 进行预测
predict(file_path)
