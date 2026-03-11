# NeoAMT

This repository includes the source codes of paper: [NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](https://arxiv.org/abs/2601.03790).

## Installation

Training Environment

```bash
# for training env
conda create --prefix ./env python=3.12
conda activate ./env

# git clone https://github.com/volcengine/verl.git
# git checkout -b new_branch 0e15c9b11c4b3f230fbc7ce75548d589635744d1
cd verl
pip3 install -e .
cd ..
pip install vllm==0.10.1.1
# pip install uv
pip install numpy==1.26.4
pip install opencv-python-headless==4.11.0.86
conda install nvidia::cuda-nvcc
pip install flash-attn==2.8.3 --no-build-isolation
pip install "ray[default]" debugpy
pip install fuzzywuzzy
pip install stanza
pip install langid
pip install jupyter
pip install -r env_requirements.txt
```

MT Scorer Environment:

```bash
# for mt metric server
conda create --prefix ./mt_score_env python=3.12
conda activate ./mt_score_env

pip install "unbabel-comet>=2.2.0"
pip install "fastapi[standard]"==0.116.1
pip install fuzzywuzzy
pip install stanza
pip install -r mt_score_env_requirements.txt

CUDA_VISIBLE_DEVICES=3 python neoamt/mt_score_server.py
```

Retrival Server Environment:

```bash
# for dense retrieval server
conda create --prefix ./retrieval_env python=3.10
conda activate ./retrieval_env
conda install -c conda-forge openjdk=22 # for using pyserini bm25
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.41.0 datasets pyserini

## install the gpu version of faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
pip install wikiextractor==3.0.6
pip install sentence-transformers
pip install "chonkie[semantic]"
pip install tokenizers==0.19.1
pip install -I mtdata==0.4.3
pip install langid
pip install numpy==1.26.4
```

## Artifacts

| Checkpoint                                                                                   |
| -------------------------------------------------------------------------------------------- |
| [NeoAMT-4B](https://huggingface.co/zhongtaomiao/NeoAMT-4B)                                   |
| [NeoAMT-4B-w-Process-Reward](zhongtaomiao/NeoAMT-4B-w-Process-Reward)                        |
| [NeoAMT-8B](https://huggingface.co/zhongtaomiao/NeoAMT-8B)                                   |
| [NeoAMT-8B-w-Process-Reward](https://huggingface.co/zhongtaomiao/NeoAMT-8B-w-Process-Reward) |

## Get Started

Download training, validation and test data:

```bash
wget https://huggingface.co/datasets/zhongtaomiao/neoamt-data/resolve/main/data.zip
unzip data.zip
```

Download wiktionary-based dictionary data from [here](https://drive.google.com/drive/folders/1r6lEBae6YAFyKUfVw7SkeXiMJzYWrGaV?usp=sharing)

Put the `retrieval_data` folder into the data folder: data/retrieval_data/wikidict/output/enwikidict_20250823/all/...

```bash


# Set up the MT Metric Server

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export HF_HOME=/lustre/miao/hf_home
conda activate /lustre/miao/neoamt/mt_score_env
python -m neoamt.mt_score_server > "mt_score_server_rqr_$(date +%Y%m%d_%H%M).log" 2>&1 &
SERVER_PID=$!
conda deactivate

# Set up the retrieval server

conda activate ./retrieval_env
LANGS="all"
file_path=data/retrieval_data/wikidict/output/enwikidict_20250823
retriever_name=bge-m3
retriever_path=/lustre/miao/models/BAAI/bge-m3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python neoamt/wiktionary_retrieval_server.py  --file_path $file_path \
                                            --topk 10 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --langs $LANGS \
                                            --faiss_gpu > "wiktionary_dense_retrieval.log" 2>&1 &
conda deactivate

# Training
conda activate ./env
ray start --head
. ./train_neoamt_8b_with_dictionary_tool.sh

# Use the following script to get the full checkpoint when the training is finished
. ./merge_model.sh

# Evaluation
# (Please run the following code after starting the MT metric and retrieval servers.)
. ./eval_neoamt_8b_with_dictionary_tool.sh # This is for generating translations and obtaining XCOMET scores.
# For llm judge eval, please read neoamt/llm_judge.sh.
# Final step is to summarize the eval results.
# For example:
python -m neoamt.summary_eval --test_output_dir output/neoamt/test-20250925_1031-neologism2plain-qwen3-8b-rqe-search-tools-enabled-True-sync_with_tools-think_search_translate-xcomet_xl-enable_thinking-True-0-1.0--1-20250927_0211

```

### Common problems:

- Token id XXX is out of vocabulary: possible solutions [link1](https://github.com/vllm-project/vllm/issues/13175) [link2](https://github.com/OpenRLHF/OpenRLHF/issues/1117) [link3](https://github.com/Simple-Efficient/RL-Factory/issues/45)

## License

This software is released under the `CC-BY-NC-SA-4.0 License`, see [LICENSE.txt](LICENSE.txt).
