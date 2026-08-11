# Financial Asset Recommendation System

A research-oriented starter project for a **deep-learning financial asset recommender**.

The system recommends stocks to synthetic investor profiles by combining:

- investor risk tolerance and sector preferences;
- historical user–asset interactions;
- asset sector and risk characteristics;
- return, volatility, valuation and dividend features.


## Research question

**Can a hybrid two-tower neural recommender produce more relevant financial-asset recommendations than non-personalized and content-based baselines?**

## Project structure

```text
Financial_Asset_Recommendation_System/
├── notebooks/
│   ├── FARS_part_1_Introduction.ipynb
│   ├── FARS_part_2_EDA.ipynb
│   ├── FARS_part_3_Dataset_Manipulation.ipynb
│   └── FARS_part_4_Experiments.ipynb
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── baselines.py
│   ├── two_tower.py
│   └── evaluation.py
├── data/raw/                 # Included reproducible starter dataset
├── data/processed/           # Created by notebook 3
├── tests/
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
jupyter notebook
```

Open and run the notebooks in numerical order.

The raw starter data is already included. To regenerate it deterministically:

```bash
python -m src.data_generation
```

## Included data

The repository contains synthetic but finance-shaped data:

- `assets.csv`: 60 fictional stocks with sector, return, volatility, beta, P/E, dividend yield and market capitalization;
- `users.csv`: 600 synthetic investors with risk profile, preferred sector, experience and horizon;
- `interactions.csv`: time-stamped views, watchlist additions and purchases.

Synthetic data keeps the starter project reproducible, legally safe and runnable offline. A later improvement can replace it with a documented public dataset while preserving the same pipeline.

## Models

1. Global popularity baseline
2. Sector-aware popularity baseline
3. Content-similarity baseline
4. Hybrid two-tower neural recommender

## Evaluation

The project uses a **temporal split** and reports:

- Hit Rate@K
- Precision@K
- Recall@K
- Mean Reciprocal Rank@K
- catalog coverage

## Mapping to the exam rubric

- **Problem statement:** Notebook 1
- **Layout and communication:** all notebooks follow one research story
- **Code quality:** reusable functions under `src/`
- **Previous research:** Notebook 1 contains references and comparison targets
- **Data gathering/cleaning:** Notebooks 2 and 3
- **Testing:** automated tests plus temporal train/validation/test evaluation
- **Visualization:** Notebook 2 and the model-comparison section in Notebook 4
