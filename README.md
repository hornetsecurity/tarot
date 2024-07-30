# TAROT

Code for the paper: **TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization**

Preprint on arXiv: coming soon

TAROT models are available on [🤗 Huggingface](https://huggingface.co/collections/gabrielloiseau/tarot-66a20fef9d0cd83041e88506).

## Installation
Clone the repository locally:
```
git clone https://github.com/hornetsecurity/tarot
cd tarot
```

Install the requirements using Poetry:
```
pip install poetry
poetry install
```

## Data
- All datasets are hosted on [🤗 Huggingface datasets](https://huggingface.co/datasets). 
- Evaluation datasets and loaded and preprocessed in `tarot/utils.py`. 
- We use the [Yelp review dataset](https://huggingface.co/datasets/Yelp/yelp_review_full) to train the generation models.

## Usage
Experimental scripts can be found in the `scripts` folder:
- `ppo-dpo.sh`: PPO and DPO fine-tuning on the Yelp dataset using policy optimization.  
- `imdb.sh`, `bac.sh`, `amt.sh`: Create for each dataset an authorship classifier and an utility classifier for evaluation, obfuscates each dataset and performs evaluation of generated text. 

Running evaluation scripts will result in the folowwing folder structure:
```
tarot
├── imdb-10
│   ├── authorship_checkpoint
│   ├── deberta-v3-authorship-imdb-10
│   ├── deberta-v3-utility-imdb-10
│   ├── utility_checkpoint
│   ├── imdb-10-test-DPO.csv
│   ├── imdb-10-test-PPO.csv
├── imdb-20
├── src
├── LICENSE
├── pyproject.toml
└── README.md
```
Where `deberta-v3-authorship-imdb-10` and `deberta-v3-utility-imdb-10` are respectively the authorship attribution and the utility classifier. `imdb-10-test-DPO.csv` and `imdb-10-test-PPO.csv` the resulting obfuscated dataset using DPO and PPO. `authorship_checkpoint` and `utility_checkpoint` are the evaluation classfier checkpoints.

## Citation
```
```
