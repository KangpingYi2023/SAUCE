import numpy as np
import pandas as pd

# 1. 读取非高斯分布的.np数据集
def load_data(file_path):
    """
    加载.np文件中的数据
    """
    data = np.load(file_path)
    return data

# 2. 计算每一列的均值和标准差
def calculate_column_stats(data):
    """
    计算每一列的均值和标准差
    """
    means = np.mean(data, axis=0)  # 按列计算均值
    std_devs = np.std(data, axis=0)  # 按列计算标准差
    return means, std_devs

# 3. 对每一列根据其均值和标准差进行高斯采样
def gaussian_sample_column(means, std_devs, num_samples):
    """
    对每一列进行高斯采样
    """
    sampled_data = np.zeros((num_samples, len(means)))  # 初始化采样数据矩阵
    for i, (mean, std_dev) in enumerate(zip(means, std_devs)):
        sampled_data[:, i] = np.random.normal(mean, std_dev, num_samples)  # 高斯采样
    return sampled_data

# 4. 将生成的每一列数据进行拼接并保存
def save_data(data, save_path):
    """
    保存生成的数据到.np文件
    """
    np.save(save_path, data)

# 主函数
def gaussian(input_file, output_file):
    data = np.load(input_file, allow_pickle=True)
    num_samples = data.shape[0]
    print("raw data 5 rows: ")
    print(data[:5])
    
    means = np.mean(data, axis=0)  
    std_devs = np.std(data, axis=0)

    sampled_data = np.zeros((num_samples, len(means)))  
    for i, (mean, std_dev) in enumerate(zip(means, std_devs)):
        sampled_data[:, i] = np.random.normal(mean, std_dev, num_samples) 
    print("Gaussian data 5 rwos: ")
    print(sampled_data[:5])

    np.save(output_file, data)

def datasets2gaussian():
    datasets=["bjaq", "census", "forest"]
    
    for dataset in datasets:
        if dataset == "census":
            input_file = f"./{dataset}/{dataset}_int.npy"  
            output_file = f"./{dataset}/{dataset}_int_gaussian.npy"  
        else:
            input_file = f"./{dataset}/{dataset}.npy"  
            output_file = f"./{dataset}/{dataset}_gaussian.npy" 

        gaussian(input_file, output_file)
        print(f"dataset {dataset} done!")

def csv_to_npy(csv_file, npy_file):
    """
    将 CSV 文件转存为 .npy 文件。

    参数:
    csv_file: str, CSV 文件路径
    npy_file: str, 输出的 .npy 文件路径
    """
    # 读取 CSV 文件
    data = pd.read_csv(csv_file)
    
    # data = data.fillna(0)

    # 将数据转换为 NumPy 数组
    numpy_array = data.to_numpy()
    
    # 保存为 .npy 文件
    np.save(npy_file, numpy_array)
    
    print(f"CSV 文件已成功转存为 {npy_file}")

# 示例调用
if __name__ == "__main__":
    tables = ["badges", "comments", "postHistory", "postLinks", "posts", "tags", "users", "votes"]

    for table in tables:
        csv_path = f"./stats/convert_data/{table}.csv"
        npy_path = f"./stats/numpy/{table}.npy"
        csv_to_npy(csv_path, npy_path)
    