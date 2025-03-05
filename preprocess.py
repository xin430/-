import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import re  # 用于正则表达式处理无效字符

def load_and_preprocess_data(file_path):
    print("Starting preprocessing...")

    # 加载数据
    df = pd.read_csv(file_path)

    # 打印数据类型，查看哪些列是字符串类型
    # print("Data types before processing:")
    # print(df.dtypes)

    # 打印数据形状，检查特征数
    # print("Original data shape:", df.shape)

    # 去除列名中的前后空格（如果有的话）
    df.columns = df.columns.str.strip()

    # 打印标签列的前几个值，检查是否有格式问题
    # print("First few labels before encoding:")
    # print(df['Label'].head(20))

    # 清理 'Label' 列的空格或换行符
    df['Label'] = df['Label'].str.strip()

    # 清理掉标签列中的非标准字符，比如多余的字符
    df['Label'] = df['Label'].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x))  # 只保留字母和数字

    # 处理 NaN 和 inf 值：仅对数值列处理
    # 替换 inf 为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 只对数值列填充 NaN
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # 打印清理后的标签列唯一值
    # print("Unique labels before encoding:", np.unique(df['Label']))

    # 将标签列转换为整数编码
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])  # 将 'BENIGN' 和其他标签转换为数字

    # 打印标签列的唯一值，确保它们已经成功从字符串转换为整数
    # print("Unique labels after encoding:", np.unique(df['Label']))

    # 选择特征列（确保选择了所有需要的特征）
    features = df.drop(columns=['Label'])

    # 打印特征列数量
    # print("Features shape:", features.shape)

    # 打印特征的列类型，确保它们都是数值类型
    # print("Feature columns data types after processing:")
    # print(features.dtypes)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 标签
    labels = df['Label']

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # 打印训练数据形状
    # print("X_train shape:", X_train.shape)

    # 将数据调整为LSTM所需的格式：样本数 × 时间步长 × 特征数
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # print("Preprocessing complete.")
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

if __name__ == "__main__":
    file_path = 'data/CICIDS_data.csv'  # 确保路径正确
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_and_preprocess_data(file_path)
