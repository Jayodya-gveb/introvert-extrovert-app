"""Introvert or Extrovert? A small Streamlit demo of a machine learning model.
Run with:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier


AUTHOR = 'ENURI_JAYODYA'                
GITHUB_URL = 'https://github.com/Jayodya-gveb/introvert-extrovert-app.git'            
LINKEDIN_URL = 'https://www.linkedin.com/in/enuri-jayodya-b78238311'    
GROUP_REPO_URL = 'https://github.com/DevDilshan/Jewelry-Sales-Prediction'        
# ------------------------------------------------------------------


def _fix(url):
    """Adds https:// if it is missing, so links never break."""
    return url if (not url or url.startswith(('http://', 'https://'))) else 'https://' + url


GITHUB_URL, LINKEDIN_URL, GROUP_REPO_URL = _fix(GITHUB_URL), _fix(LINKEDIN_URL), _fix(GROUP_REPO_URL)

st.set_page_config(page_title='Introvert or Extrovert?', page_icon='🧠', layout='centered')

VIOLET, ORANGE = '#8b5cf6', '#f97316'
NAMES = {
    'Time_spent_Alone': 'Time spent alone',
    'Stage_fear': 'Stage fear',
    'Social_event_attendance': 'Social event attendance',
    'Going_outside': 'Going outside',
    'Drained_after_socializing': 'Drained after socializing',
    'Friends_circle_size': 'Friends circle size',
    'Post_frequency': 'Social media posting',
}

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 760px;}
.hero {background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #c2410c 100%);
       padding: 28px 26px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: 6px 0 0 0; color: #ede9fe; font-size: 1rem;}
.card {background: #1a1432; border: 1px solid #2e2457; border-radius: 14px; padding: 18px 20px; margin: 14px 0;}
.card .small {color: #a8a0c9; font-size: .8rem; letter-spacing: .1em; text-transform: uppercase;}
.card .big {font-size: 2rem; font-weight: 800; color: #fff; margin: 2px 0 8px 0;}
.chip {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem; font-weight: 700; color: #0f0b1e;}
.spectrum {position: relative; height: 16px; border-radius: 999px; margin: 18px 0 6px 0;
           background: linear-gradient(90deg, #8b5cf6, #f97316);}
.marker {position: absolute; top: -6px; width: 6px; height: 28px; background: #fff; border-radius: 3px;
         box-shadow: 0 0 0 2px #0f0b1e; transform: translateX(-3px);}
.ends {display: flex; justify-content: space-between; color: #cbd5e1; font-size: .85rem;}
.crow {display: flex; align-items: center; margin: 7px 0;}
.cname {width: 44%; font-size: .9rem; color: #ede9fe;}
.ctrack {position: relative; width: 56%; height: 14px; background: #241c45; border-radius: 7px;}
.ctrack:after {content: ''; position: absolute; left: 50%; top: -3px; width: 1px; height: 20px; background: #6b5fa5;}
.cbar {position: absolute; top: 0; height: 14px; border-radius: 7px;}
.note {color: #a8a0c9; font-size: .85rem; margin-top: 8px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = XGBClassifier()
    model.load_model('assets/model.json')
    with open('assets/meta.json') as f:
        meta = json.load(f)
    return model, meta


try:
    model, meta = load_assets()
except Exception:
    st.error('Model files not found. Run  python build_model.py  first (see README.md).')
    st.stop()

R = meta['ranges']
FEATURES = meta['features']

st.markdown('<div class="hero"><h1>🧠 Introvert or Extrovert?</h1>'
            '<p>Answer 7 quick questions. A machine learning model estimates which side you lean towards.</p></div>',
            unsafe_allow_html=True)

tab_quiz, tab_how, tab_model = st.tabs(['Quiz', 'How it works', 'About the model'])

with tab_quiz:
    c1, c2 = st.columns(2)
    alone = c1.slider('Time spent alone each day (hours)', R['Time_spent_Alone'][0], R['Time_spent_Alone'][1],
                      R['Time_spent_Alone'][2])
    events = c2.slider('How often you attend social events', R['Social_event_attendance'][0],
                       R['Social_event_attendance'][1], R['Social_event_attendance'][2],
                       help='0 = never, 10 = very often')
    outside = c1.slider('How often you go outside', R['Going_outside'][0], R['Going_outside'][1],
                        R['Going_outside'][2], help='0 = rarely, higher = more often')
    friends = c2.slider('Size of your friends circle', R['Friends_circle_size'][0], R['Friends_circle_size'][1],
                        R['Friends_circle_size'][2])
    posts = c1.slider('How often you post on social media', R['Post_frequency'][0], R['Post_frequency'][1],
                      R['Post_frequency'][2], help='0 = never, 10 = very often')
    fear = c2.radio('Do you have stage fear?', ['No', 'Yes'], horizontal=True)
    drained = c1.radio('Do you feel drained after socialising?', ['No', 'Yes'], horizontal=True)

    X = pd.DataFrame([{
        'Time_spent_Alone': alone,
        'Stage_fear': int(fear == 'Yes'),
        'Social_event_attendance': events,
        'Going_outside': outside,
        'Drained_after_socializing': int(drained == 'Yes'),
        'Friends_circle_size': friends,
        'Post_frequency': posts,
    }])[FEATURES]

    p_intro = float(model.predict_proba(X)[0][1])        # class 1 = Introvert
    p_extro = 1 - p_intro
    lean = 'Introvert' if p_intro >= 0.5 else 'Extrovert'
    top = max(p_intro, p_extro)
    if top >= 0.9:
        label, color = 'Strong lean', '#22c55e'
    elif top >= 0.7:
        label, color = 'Moderate lean', '#f59e0b'
    else:
        label, color = 'Borderline: close to the middle', '#ef4444'

    pos = min(max(p_extro * 100, 2), 98)
    st.markdown(
        f'<div class="card"><div class="small">The model says you lean towards</div>'
        f'<div class="big">{lean} ({top * 100:.0f}%)</div>'
        f'<span class="chip" style="background:{color}">{label}</span>'
        f'<div class="spectrum"><div class="marker" style="left:{pos:.1f}%"></div></div>'
        f'<div class="ends"><span>Introvert</span><span>Extrovert</span></div></div>',
        unsafe_allow_html=True)

    # why: per-feature contributions (built into XGBoost). Positive = pushes towards Introvert.
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    push_extro = pd.Series(-contrib, index=FEATURES)
    order = push_extro.abs().sort_values(ascending=False).index
    biggest = max(push_extro.abs().max(), 1e-9)
    rows = ''
    for f in order:
        v = push_extro[f]
        w = abs(v) / biggest * 48
        left = 50 if v >= 0 else 50 - w
        col = ORANGE if v >= 0 else VIOLET
        rows += (f'<div class="crow"><div class="cname">{NAMES[f]}</div><div class="ctrack">'
                 f'<div class="cbar" style="left:{left:.1f}%;width:{w:.1f}%;background:{col}"></div></div></div>')
    st.markdown(f'<div class="card"><div class="small">Why? What pushed the result</div>{rows}'
                f'<div class="ends" style="margin-top:6px"><span>towards Introvert</span><span>towards Extrovert</span></div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown('<div class="note">This is a fun demo of machine learning, not a psychological test. '
                'Real personality is not a switch between two types, so treat the result as a rough guide.</div>',
                unsafe_allow_html=True)

with tab_how:
    st.markdown(f"""
**What it does.** You answer 7 questions about everyday habits. A model trained on a survey-style dataset of
{meta['n_rows']:,} people estimates how much your answers look like those of people labelled Introvert or Extrovert.

**The 7 inputs.** Time spent alone, stage fear, social event attendance, going outside, feeling drained after
socialising, friends circle size, and how often you post online.

**Preparing the data.** Yes/No answers are turned into 1 and 0 so the computer can use them. The model is an
XGBoost classifier (many small decision trees, each correcting the mistakes of the one before).

**The explanation bars.** Each bar shows how much one of your answers pushed the result towards Introvert (left,
violet) or Extrovert (right, orange), calculated with SHAP-style contributions built into XGBoost.
""")

with tab_model:
    st.markdown('**Model comparison** (5-fold cross-validation, duplicate rows removed)')
    cv = pd.DataFrame(meta['cv']).T
    cv = cv.rename(columns={'accuracy': 'Accuracy', 'f1': 'F1-score', 'roc_auc': 'ROC AUC'})[['Accuracy', 'F1-score', 'ROC AUC']]
    st.table(cv.round(3))
    st.markdown(f"""
**Honest reading.** All three models score about 99%, and even the simplest one (Logistic Regression) is close behind.
When a basic model does this well, the data is usually very clean or artificial: the labels follow strong, simple patterns
that real people do not. So the high score shows the model learned this dataset, **not** that it can measure real
personality. The dataset has {meta['n_rows']:,} rows ({meta['n_rows'] - meta['n_unique_rows']} duplicates) with an even
split between the two labels.
""")

st.divider()
links = []
if GITHUB_URL:
    links.append(f'[GitHub]({GITHUB_URL})')
if LINKEDIN_URL:
    links.append(f'[LinkedIn]({LINKEDIN_URL})')
who = f'Built by {AUTHOR}. ' if AUTHOR else ''
origin = f' Based on a [group project]({GROUP_REPO_URL}).' if GROUP_REPO_URL else ''
st.caption(f"{who}{' · '.join(links)}{origin}  \nEducational demo. Not a psychological assessment.")
