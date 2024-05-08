exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

python "${work_dir}"/bins/tts/preprocess.py --config=/home/srt15/amphion/egs/tts/NaturalSpeech2/exp_config.json