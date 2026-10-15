"""Train the personality model and save it for the app. Run once:  python build_model.py

Reads  data/personality_dataset.csv
Writes assets/model.json and assets/meta.json
"""
import json
import os

import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

os.makedirs('assets', exist_ok=True)

df = pd.read_csv('data/personality_dataset.csv').dropna()
df['Stage_fear'] = (df['Stage_fear'] == 'Yes').astype(int)
df['Drained_after_socializing'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
y = (df['Personality'] == 'Introvert').astype(int)      # 1 = Introvert, 0 = Extrovert
X = df.drop(columns='Personality')

# ---- honest comparison: 5-fold stratified CV on rows with duplicates removed ----
keep = ~X.duplicated()
Xd, yd = X[keep], y[keep]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
candidates = {
    'Logistic Regression': Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             random_state=42, eval_metric='logloss'),
}
cv_table = {}
for name, mdl in candidates.items():
    r = cross_validate(mdl, Xd, yd, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
    cv_table[name] = {'accuracy': float(r['test_accuracy'].mean()), 'accuracy_std': float(r['test_accuracy'].std()),
                      'f1': float(r['test_f1'].mean()), 'roc_auc': float(r['test_roc_auc'].mean())}
    print(f"{name:20s} accuracy {cv_table[name]['accuracy']:.3f} +/- {cv_table[name]['accuracy_std']:.3f}")

# ---- final model: XGBoost (the model chosen in the group project), fitted on all rows ----
model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
model.fit(X, y)
model.save_model('assets/model.json')      # JSON format: works across xgboost versions

meta = {
    'features': list(X.columns),
    'ranges': {c: [int(X[c].min()), int(X[c].max()), int(X[c].median())] for c in X.columns},
    'cv': cv_table,
    'n_rows': int(len(df)),
    'n_unique_rows': int(keep.sum()),
    'class_share': {k: float(v) for k, v in df['Personality'].value_counts(normalize=True).items()},
    'xgboost_version': xgboost.__version__,
}
with open('assets/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Saved assets/model.json and assets/meta.json')
