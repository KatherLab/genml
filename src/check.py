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

