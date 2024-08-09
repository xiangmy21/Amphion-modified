import json
import os

# 设置train.json和TextGrid文件夹的路径
base_path = '/mnt/nvme_share/test004/amphion/data/LJSpeech'

ignore_first_part = False # 是否忽略uid第一个#前内容（建立文件夹时）

train_json_path = f'{base_path}/train.json'  # train.json文件的路径
textgrid_folder_path = f'{base_path}/TextGrid'  # TextGrid文件夹的根目录
duration_folder_path = f'{base_path}/duration'
phone_folder_path = f'{base_path}/phone'
pitch_folder_path = f'{base_path}/pitch_avg'


# 读取train.json文件
with open(train_json_path, 'r') as file:
    data = json.load(file)

# 新的数据列表，只包含存在对应数据文件的条目
new_data = []

# 检查每个条目的数据文件是否存在
for entry in data:

    # 解析Uid以找到对应的TextGrid文件路径
    uid_parts = entry['Uid'].split('#')
    
    # 根据开关决定是否忽略第一个部分
    if ignore_first_part:
        uid_replaced = '/'.join(uid_parts[1:])
    else:
        uid_replaced = '/'.join(uid_parts)
    textgrid_path = os.path.join(textgrid_folder_path, f"{uid_replaced}.TextGrid")

    duration_path = os.path.join(duration_folder_path, entry['Uid']+'.npy')
    phone_path = os.path.join(phone_folder_path, entry['Uid']+'.txt')
    pitch_path = os.path.join(pitch_folder_path, entry['Uid']+'.npy')
    
    # 检查文件是否存在
    if os.path.exists(textgrid_path) and \
        os.path.exists(duration_path) and \
        os.path.exists(phone_path) and \
        os.path.exists(pitch_path):
        new_data.append(entry)

# 将存在数据文件的条目保存到new_train.json中
new_train_json_path = f'{base_path}/train_clean.json'  # 新的train.json文件路径
with open(new_train_json_path, 'w') as file:
    json.dump(new_data, file, indent=4)

print(f"新的train.json已保存至{new_train_json_path}，共{len(new_data)}条有效数据。")