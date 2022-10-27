# framing-police-violence

Authors: [Caleb Ziems](calebziems.com), [Diyi Yang](diyiyang.com)

This repository contains data links and code for the paper:
> Ziems, C. & Yang, D. (2021). To Protect and To Serve? Analyzing Entity-Centric Framing of Police Violence In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)_.

```
@inproceedings{ziems2021protect,
 author = {Ziems, Caleb and Yang, Diyi},
 booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
 title = {{To Protect and To Serve? Analyzing Entity-Centric Framing of Police Violence}},
 year = {2021}
}
```

## Prerequisites - Environment
* [anaconda](https://www.anaconda.com/products/individual)
Create main project environment
```bash
conda create --name framing-pv python=3.7
conda activate dragnet
pip install -r requirements_dragnet.txt
conda deactivate

conda create --name coref python=3.7
conda activate coref
pip install -r requirements_coref.txt
conda deactivate

conda create --name framing-pv python=3.7
conda activate framing-pv
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Project Pipeline

All data is contained in a zip file in the [Drive directory](https://drive.google.com/file/d/12Kg0PS-kx6q1NR-Vwdfj7cM4yb3LrMqE/view?usp=sharing)

1. Download all data and setup repo by running `bash populate_repo.sh`

2. Run `python 01_pull_shooting_articles.py` to scrape news articles on police killings

3. Clean the retrieved articles by first switching to `conda activate dragnet` and running `python 03_dragnet_clean.py --input_glob "data/raw/shootings-articles/*/*.html" --output "data/raw/shootings-txt"`

4. Return to `conda activate framing-pv` and compile all scraped shooting articles with their political leanings by running `python 04_build_shooting_df.py`

5. Switch to `conda activate coref` and extract all frames by running `python 05_framing_functions.py`

6. Switch back to `conda activate framing-pv` and run `python 06_clean_framing_file.py` to generate the composite file for framing analysis

7. Run the analyses in `paper-analysis.ipynb` and `protest-granger.ipynb`
