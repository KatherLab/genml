import torch

# 加载 .pt 文件
feature_path = '/mnt/bulk-io/lizhang/LiWorkSpace/genomics/data/04_feature/dnabert2_stack_True_cls_False/TCGA-02-0003.pt'
features = torch.load(feature_path)

# 检查加载的特征内容
if isinstance(features, dict):
    # 如果保存的是一个字典
    for key, value in features.items():
        print(f"Key: {key}, Value: {value.shape if isinstance(value, torch.Tensor) else type(value)}")
elif isinstance(features, list):
    # 如果保存的是一个列表
    for idx, item in enumerate(features):
        print(f"Index: {idx}, Value: {item.shape if isinstance(item, torch.Tensor) else type(item)}")
elif isinstance(features, torch.Tensor):
    # 如果保存的是一个单独的 Tensor
    print(f"Tensor Shape: {features.shape}")
else:
    print(f"Unknown data type: {type(features)}")

# 示例：打印特定特征的数据
#print(features)  # 或者特定的 key: print(features['some_key'])

#%%

import transformers
print(transformers.__version__)


# %%
import pandas as pd

def load_data(file_path: str, pat_column: str, num_patients: int = None) -> pd.DataFrame:
    print('num_patients:', num_patients)
    data = pd.read_csv(file_path)
    if num_patients is not None:
        data = data[data[pat_column].isin(data[pat_column].unique()[:num_patients])]
    return data

file_path = '/mnt/bulk-io/lizhang/LiWorkSpace/genomics/data/01_raw/tcga_mutations_controlled.csv'
pat_column = "Patient_ID"
mut_column = "Alt_Sequence"

raw_data = load_data(file_path, pat_column)

# 筛选特定病人的数据
patient_data = raw_data[raw_data[pat_column] == 'TCGA-02-0047']

# 计算 Alt_Sequence 列的每个元素的长度
patient_data['Alt_Sequence_len'] = patient_data[mut_column].apply(len)

# 显示结果
print(patient_data[['Patient_ID', 'Alt_Sequence', 'Alt_Sequence_len']])
# %%
