import torch
from model import LSTM_Model  # 导入你的模型结构
from sklearn.metrics import accuracy_score
import numpy as np

# 初始化模型（与训练时相同的参数）
input_size = 78  # 特征数量
hidden_size = 128
num_classes = 8  # 类别数量（根据你的任务修改）
model = LSTM_Model(input_size, hidden_size, num_classes)

# 加载保存的模型权重
model.load_state_dict(torch.load('lstm_model.pth'))  # 加载训练时保存的模型权重
model.eval()  # 切换到评估模式（推理模式）

# 假设你已经有预处理好的测试数据 X_test_tensor 和 y_test_tensor
# 如果没有，可以从你的数据加载和预处理步骤中获得这些数据

# 以下是一个简单的示例：
# 1. 假设 X_test_tensor 和 y_test_tensor 已经准备好了
# 2. 你可以将其替换为你的真实测试数据。

# 例如，假设 X_test_tensor 和 y_test_tensor 是你的测试数据
# X_test_tensor = <你的测试数据输入>  # 形状应该是 [batch_size, seq_len, input_size]
# y_test_tensor = <你的测试标签>  # 形状应该是 [batch_size]

# 进行预测
with torch.no_grad():  # 禁用梯度计算，节省内存
    outputs = model(X_test_tensor)  # 将测试数据传入模型

# 使用 torch.max 获得每个样本的预测类别
_, predicted = torch.max(outputs, 1)  # 预测类别是最大分数的索引

# 输出预测标签
print("Predicted labels:", predicted)

# 计算准确率
# 将预测的标签与真实标签进行比较
accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())  # 计算准确率
print(f"Test Accuracy: {accuracy:.4f}")
