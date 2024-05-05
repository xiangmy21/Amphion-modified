import os
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import defaultdict

output_path = "/home/srt15/hs/LatentDiffusion/amphion/data/libritts"
dataset_path = "/home/srt15/hs/LatentDiffusion/amphion/LibriTTSLabel-master/lab/phone"
phone_path = os.path.join(output_path, "phone")
os.makedirs(phone_path, exist_ok=True)
duration_path = os.path.join(output_path, "duration")
os.makedirs(duration_path, exist_ok=True)

def handleFile(file_path, uid):
    # 读取源文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 准备两个空列表来储存音素和持续时间
    phonemes = []
    durations = []

    # 处理每一行
    for line in lines:
        # 分割行
        parts = line.split()

        # 如果行中有三个部分（开始时间、结束时间、音素）
        if len(parts) == 3:
            start_time, end_time, phoneme = parts
            # 将音素添加到列表
            phonemes.append(phoneme)
            # 计算持续时间并添加到列表
            duration = float(end_time) - float(start_time)
            durations.append(duration)

    # 创建音素文件
    with open(os.path.join(phone_path, uid+".phone"), 'w') as f:
        f.write(' '.join(phonemes))

    # 创建持续时间文件
    np.save(os.path.join(duration_path, uid+".npy"), durations)

def libritts_statistics(data_dir):
    speakers = []
    distribution2speakers2pharases2utts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    distribution_infos = glob(data_dir + "/*")

    for distribution_info in distribution_infos:
        distribution = distribution_info.split("/")[-1]
        print(distribution)
        if distribution != "train-clean-100":
            continue

        speaker_infos = glob(distribution_info + "/*")

        if len(speaker_infos) == 0:
            continue

        for speaker_info in speaker_infos:
            speaker = speaker_info.split("/")[-1]

            speakers.append(speaker)

            pharase_infos = glob(speaker_info + "/*")

            for pharase_info in pharase_infos:
                pharase = pharase_info.split("/")[-1]

                utts = glob(pharase_info + "/*.lab")

                for utt in utts:
                    uid = utt.split("/")[-1].split(".")[0]
                    distribution2speakers2pharases2utts[distribution][speaker][
                        pharase
                    ].append(uid)

    unique_speakers = list(set(speakers))
    unique_speakers.sort()

    print("Speakers: \n{}".format("\t".join(unique_speakers)))
    return distribution2speakers2pharases2utts, unique_speakers

def main():
    print("-" * 10)
    print("Preparing samples for libritts...\n")

    # Load
    libritts_path = dataset_path


    distribution2speakers2pharases2utts, _ = libritts_statistics(
        libritts_path
    )

    for distribution, speakers2pharases2utts in tqdm(
        distribution2speakers2pharases2utts.items()
    ):
        for speaker, pharases2utts in tqdm(speakers2pharases2utts.items()):
            pharase_names = list(pharases2utts.keys())

            for chosen_pharase in pharase_names:
                for chosen_uid in pharases2utts[chosen_pharase]:
                    uid = "{}#{}#{}#{}".format(
                            distribution, speaker, chosen_pharase, chosen_uid
                        )
                    lab_path = os.path.join(dataset_path, "{}/{}/{}/{}.lab".format(
                        distribution, speaker, chosen_pharase, chosen_uid
                    ))
                    handleFile(lab_path, uid)
                    
    
if __name__ == "__main__":
    # code = np.load("/home/srt15/amphion/data/libritts/code/train-clean-100#19#198#19_198_000000_000000.npy")
    # print(code[0].shape)
    main()