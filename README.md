# Introvert or Extrovert? (Streamlit demo)

A small web app that estimates whether answers to 7 everyday-habit questions look more like an Introvert or an Extrovert,
using an XGBoost model. Educational demo, not a psychological test.

Based on a university group project (Introvert-or-Extrovert-Prediction).

## Files
```
app.py                    the web app
build_model.py            trains the model and saves it (run once)
requirements.txt
data/personality_dataset.csv   the survey-style dataset (5,000 rows)
assets/                   created by build_model.py
.streamlit/config.toml    theme
```

## Run locally
1. pip install -r requirements.txt
2. python build_model.py
3. streamlit run app.py      (opens http://localhost:8501)

## Notes
- All models reach about 99% in cross-validation, even Logistic Regression, which suggests the dataset is very clean or
  artificial. The score shows the model learned this dataset, not that it can measure real personality.
