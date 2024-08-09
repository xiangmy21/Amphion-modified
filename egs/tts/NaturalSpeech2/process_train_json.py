import json

train_file_path = "/mnt/nvme_share/test004/amphion/data/libritts/train.json"
missing_file_path = "/mnt/nvme_share/test004/LibriTTSLabel-master/missing_files.txt"
new_train_file_path = "/mnt/nvme_share/test004/amphion/data/libritts/train_miss.json"

# 读取并处理 B 文件
with open(missing_file_path, 'r') as f:
    missing_entries = {line.strip().split('/')[-1] for line in f}

print(missing_entries)

# 读取 A 文件
with open(train_file_path, 'r') as f:
    data = json.load(f)

# 保留不在 B 文件中的条目
new_data = [entry for entry in data if entry['Uid'].split('#')[-1] not in missing_entries]

# 写入新的 JSON 文件
with open(new_train_file_path, 'w') as f:
    json.dump(new_data, f, indent=4)