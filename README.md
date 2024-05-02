<h1 align="center">TacoLM: GaTed Attention Equipped Codec Language Model are Efficient Zero-Shot Text to Speech Synthesizers</h1>

This is the PyTorch implementation of the TacoLM paper. This repository is based on the [fairseq package v0.12.2](https://github.com/pytorch/fairseq/tree/v0.12.2). [Code](https://anonymous.4open.science/r/TacoLM)

# Requirements and Installation

It's recommended to use conda to create environments and install packages. 
``` bash
conda create -n TacoLM python=3.9
conda activate TacoLM
```

Then download the version of [pytorch](https://pytorch.org/get-started/previous-versions/) that matches your OS and cuda version. Here is an example.
``` bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Finally, install TacoLM from this repository.
``` bash
pip install -e .
```

Other dependencies:
``` bash
pip install -e git+https://git@github.com/facebookresearch/encodec#egg=encodec
pip install transformers
pip install jiwer
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
pip install soundfile
pip install Cython
pip install nemo_toolkit['all']==1.20.0
pip install tensorboardX
pip install pyarrow
```

# Training

## Data Preparing
Please follow README in the `preprocess_librispeech` folder.

## AR model
``` bash
WANDB_NAME=TacoLM_AR \
fairseq-train ${data_split_path} \
--task language_modeling --preload-dataset --modified --ar-task  -a mega_valle_09 --save-dir checkpoints/TacoLM_AR --skip-invalid-size-inputs-valid-test --skip-remainder-batch --max-target-positions 4096 --normalize-before --no-affine-final-norm --max-tokens 8192 --tokens-per-sample 8192 --update-freq 2 --truncation-length 25600 --rel-pos-bias rotary --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-3 --clip-norm 0.0 --lr-scheduler linear_decay --total-num-update 300000 --max-update 300000 --end-learning-rate 0.0 --warmup-updates 15000 --warmup-init-lr 1e-07 --criterion cross_entropy --dropout 0.1 --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.01 --valid-block splits:10 --fp16 --wandb-project TacoLM --num-workers 3
```

## NAR model
``` bash
WANDB_NAME=TacoLM_AR \
fairseq-train ${data_split_path} \
--task language_modeling --preload-dataset --modified -a mega_valle_09 --save-dir checkpoints/TacoLM_NAR  --skip-invalid-size-inputs-valid-test --skip-remainder-batch --max-target-positions 4096 --normalize-before --no-affine-final-norm --max-tokens 8192 --tokens-per-sample 8192 --update-freq 2 --truncation-length 25600 --rel-pos-bias rotary --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-3 --clip-norm 0.0 --lr-scheduler linear_decay --total-num-update 300000 --max-update 300000 --end-learning-rate 0.0 --warmup-updates 15000 --warmup-init-lr 1e-07 --criterion cross_entropy --dropout 0.1 --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.01 --valid-block splits:10 --fp16 --wandb-project TacoLM --num-workers 3
```

## Speech Synthesis
``` bash
python eval_code/generate_all.py
```

## Speech Evaluation
``` bash
python eval_code/asr.py
```


# License

This repository is under MIT license.
