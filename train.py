import torch
import torch.optim as optim
import torch.nn as nn  # 确保导入 nn 模块
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from model import LSTM_Model
from preprocess import load_and_preprocess_data
import numpy as np

# 超参数设置
input_size = 78  # 根据数据集特征数设置
hidden_size = 128
num_classes = 8  # 假设有5个攻击类型
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# 数据路径
file_path = 'data/CICIDS_data.csv'

# 加载和预处理数据
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_and_preprocess_data(file_path)

# 在训练前查看标签
print("Unique labels in training data:", np.unique(y_train_tensor))

# 创建DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 创建模型
model = LSTM_Model(input_size, hidden_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss 进行多分类
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 选择设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 打印输入数据的形状，确保正确
        print(f"Inputs shape: {inputs.shape}")

        # 前向传播
        outputs = model(inputs)
        print(f"outputs shape: {outputs.shape}")
        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    # 每个epoch结束后打印训练损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# 测试模型
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 打印输入数据的形状，确保正确
        print(f"Test inputs shape: {inputs.shape}")

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')
