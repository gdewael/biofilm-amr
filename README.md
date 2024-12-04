# biofilm-amr
Code supporting "Predicting the antibiotic susceptibility of Pseudomonas aeruginosa biofilms using machine learning models".

## Quick start

All code has been tested on a linux system using python 3.10.
We recommend running our scripts in a fresh conda environment. For example:
```bash
git clone https://github.com/gdewael/biofilm-amr.git
conda create --name biofilmamr python=3.10
conda activate biofilmamr
pip install -r ./biofilm-amr/requirements.txt
```

To fully reproduce all results, run:
```bash
cd ./biofilm-amr
bash ./reproduce.sh
```

## Repository contents

- `./data/evolved/` Folder containing preprocessed data `.csv` files derived from experimentally evolved samples.
- `./data/isolates/` Folder containing preprocessed data `.csv` files derived from clinical isolates.
- `./scripts/` Folder containing all individual python scripts for every machine learning experiment.
- `./reproduce.sh` A helper bash script that runs all experiments sequentially.