import json
import os

# 设置train.json和TextGrid文件夹的路径
train_json_path = '/home/srt15/amphion/data/libritts/train_Final_may.json'  # train.json文件的路径
textgrid_folder_path = '/home/srt15/amphion/data/libritts/TextGrid'  # TextGrid文件夹的根目录

# 读取train.json文件
with open(train_json_path, 'r') as file:
    data = json.load(file)

# 新的数据列表，只包含存在对应TextGrid文件的条目
new_data = []

# 检查每个条目的TextGrid文件是否存在
for entry in data:
    # 解析Uid以找到对应的TextGrid文件路径
    uid_parts = entry['Uid'].split('#')
    singer_id = uid_parts[1]
    chapter_id = uid_parts[2]
    textgrid_path = os.path.join(textgrid_folder_path, singer_id, chapter_id, f"{uid_parts[3]}.TextGrid")
    
    # 检查文件是否存在
    if os.path.exists(textgrid_path):
        new_data.append(entry)

# 将存在TextGrid文件的条目保存到new_train.json中
new_train_json_path = '/home/srt15/amphion/data/libritts/train_final_clean.json'  # 新的train.json文件路径
with open(new_train_json_path, 'w') as file:
    json.dump(new_data, file, indent=4)

print(f"新的train.json已保存至{new_train_json_path}，共{len(new_data)}条有效数据。")