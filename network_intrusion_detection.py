import torch.optim as optim
from sklearn.metrics import accuracy_score

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于多分类的交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 遍历训练集
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        # 前向传播
        outputs = model(inputs)

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

    # 打印每个epoch的训练损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# 测试模型
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 对新数据进行预测（假设X_new为待预测的特征数据）
X_new = X_test_tensor[:10]  # 示例：对测试集的前10个样本进行预测
model.eval()
with torch.no_grad():
    predictions = model(X_new)

# 将预测结果转换为类别标签
predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()

# 打印部分预测结果
print(predicted_labels)
