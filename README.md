This is the PyTorch implementation of the VALLE paper. This folder is based on the [fairseq package v0.12.2](https://github.com/pytorch/fairseq/tree/v0.12.2).

# Requirements and Installation

It's recommended to use conda to create environments and install packages. 
``` bash
conda create -n valle python=3.9
conda activate valle
```

Then download the version of [pytorch](https://pytorch.org/get-started/previous-versions/) that matches your OS and cuda version. Here is an example.
``` bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Finally, install valle from this repository.
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
#WANDB_NAME=valle \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
fairseq-train ${data_split_path} \
--save-dir checkpoints/valle \
#--restore-file checkpoints/valle/checkpoint10.pt \
#--wandb-project valle \
--task language_modeling --modified -a transformer_lm \
--skip-invalid-size-inputs-valid-test --skip-remainder-batch --max-target-positions 4096 \
--optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 32000 --warmup-init-lr 1e-07 \
--tokens-per-sample 4096 --max-tokens 4096 --update-freq 4 \
--fp16 --max-update 200000 --num-workers 3 \
--ar-task --text-full-attention \
--n-control-symbols 80
```

## NAR model
``` bash
#WANDB_NAME=valle-nar \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
fairseq-train ${data_split_path} \
--save-dir checkpoints/valle-nar \
#--wandb-project valle-nar \
--task language_modeling --modified -a transformer_lm \
--skip-invalid-size-inputs-valid-test --skip-remainder-batch --max-target-positions 4096 \
--optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 32000 --warmup-init-lr 1e-07 \
--tokens-per-sample 4096 --max-tokens 4096 --update-freq 3 \
--fp16 --max-update 200000 --num-workers 3 \
--n-control-symbols 80
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
