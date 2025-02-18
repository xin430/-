import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 多分类输出层

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])  # 取最后一个时间步的输出
        return out
