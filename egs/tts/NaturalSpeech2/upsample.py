import torchaudio
from torchaudio.transforms import Resample
import os
from tqdm import tqdm

# 定义路径
input_dir = '/mnt/nvme_share/test004/LJSpeech'
output_dir = '/mnt/nvme_share/test004/LJSpeech_24k'
os.makedirs(output_dir, exist_ok=True)

# 定义上采样的转换器
resample = Resample(orig_freq=22050, new_freq=24000)

# 遍历所有文件并进行上采样
for root, _, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            waveform, sample_rate = torchaudio.load(file_path)
            upsampled_waveform = resample(waveform)
            
            # 保存到新目录
            output_file_path = os.path.join(output_dir, file)
            torchaudio.save(output_file_path, upsampled_waveform, 24000)

print("上采样完成！")
