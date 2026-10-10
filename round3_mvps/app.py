"""
Dashboard for the SDS (personality -> success) and JDS (skills -> salary hike) models.
Run:  streamlit run app.py
Keep these 4 files in the same folder as app.py:
  SDS Personality Traits.xlsx, JDS Skill Traits.xlsx,
  SDS_Personality_Success_Model.pkl, JDS_Salary_Hike_Model.pkl
"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent

CONFIGS = {
    "SDS: Personality → Success": dict(
        xlsx="SDS Personality Traits.xlsx",
        pkl="SDS_Personality_Success_Model.pkl",
        target_key="success",
        labels={
            "neuroticism": "Neuroticism",
            "extraversion": "Extraversion",
            "openness_to_experience": "Openness",
            "agreeableness": "Agreeableness",
            "conscientiousness": "Conscientiousness",
        },
        lo=0.0, hi=100.0, step=1.0,
        class_names={0: "Low Success", 1: "High Success"},
        input_help="Enter your SDS questionnaire trait scores (0 – 100).",
    ),
    "JDS: Skills → Salary Hike": dict(
        xlsx="JDS Skill Traits.xlsx",
        pkl="JDS_Salary_Hike_Model.pkl",
        target_key="salary_hike",
        labels={
            "big_data_skills": "Big Data",
            "maths_stats_skills": "Maths & Stats",
            "coding_skills": "Coding",
            "ai_and_ml_skills": "AI & ML",
            "dashboard_and_storytelling_skills": "Dashboard & Storytelling",
        },
        lo=1.0, hi=5.0, step=0.1,
        class_names={0: "Low Hike", 1: "High Hike"},
        input_help="Rate each skill from 1.0 (very low) to 5.0 (excellent).",
    ),
}


# ---------------------------------------------------------------- loaders
def clean_columns(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace("-", "_").replace("__", "_")
                  for c in df.columns]
    return df


@st.cache_data
def load_df(name):
    return clean_columns(pd.read_excel(BASE / name))


@st.cache_resource
def load_pack(name):
    return joblib.load(BASE / name)


def positive_index(classes):
    classes = list(classes)
    return classes.index(1) if 1 in classes else len(classes) - 1


# ---------------------------------------------------------------- page
def render(cfg):
    missing = [f for f in (cfg["xlsx"], cfg["pkl"]) if not (BASE / f).exists()]
    if missing:
        st.error(f"Missing file(s) next to app.py: {', '.join(missing)}")
        st.stop()

    df = load_df(cfg["xlsx"])
    pack = load_pack(cfg["pkl"])
    model, scaler = pack["model"], pack.get("scaler")
    feats = list(pack["features"])
    names = cfg["class_names"]
    nice = cfg["labels"]

    target = next(c for c in df.columns if cfg["target_key"] in c.lower())
    df["Outcome"] = df[target].map(lambda v: names.get(v, str(v)))
    ordered = [names.get(c, str(c)) for c in sorted(df[target].dropna().unique())]

    def prep(A):
        A = A[feats]
        return scaler.transform(A) if scaler is not None else A

    tab1, tab2, tab3 = st.tabs(["📊 Data analysis", "🎯 Model performance", "🧪 Try it yourself"])

    # ---------- Data analysis
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Missing values", int(df[feats + [target]].isna().sum().sum()))
        c3.metric("Duplicate rows", int(df.duplicated().sum()))
        st.dataframe(df.drop(columns="Outcome").head(50), use_container_width=True)

        left, right = st.columns(2)
        with left:
            counts = df["Outcome"].value_counts().reindex(ordered).reset_index()
            counts.columns = ["Outcome", "Count"]
            st.plotly_chart(px.bar(counts, x="Outcome", y="Count", color="Outcome",
                                   title="Class distribution"), use_container_width=True)
        with right:
            corr_df = df[feats].copy()
            corr_df[target] = pd.to_numeric(df[target], errors="coerce")
            corr_df = corr_df.rename(columns=nice)
            st.plotly_chart(px.imshow(corr_df.corr(), text_auto=".2f", aspect="auto",
                                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                      title="Correlation heatmap"), use_container_width=True)

        means = df.groupby("Outcome")[feats].mean().rename(columns=nice)
        means = means.reset_index().melt(id_vars="Outcome", var_name="Feature", value_name="Average")
        st.plotly_chart(px.bar(means, x="Feature", y="Average", color="Outcome", barmode="group",
                               title="Average score by outcome"), use_container_width=True)

        f = st.selectbox("Explore one feature", feats, format_func=lambda x: nice.get(x, x))
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(df, x=f, color="Outcome", barmode="overlay", opacity=0.7,
                                    title="Distribution"), use_container_width=True)
        b.plotly_chart(px.box(df, x="Outcome", y=f, color="Outcome", title="Box plot"),
                       use_container_width=True)

    # ---------- Model performance
    with tab2:
        X, y = df[feats], df[target]
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        y_pred = model.predict(prep(X_te))
        acc = accuracy_score(y_te, y_pred)

        m1, m2 = st.columns(2)
        m1.metric("Test accuracy (recomputed)", f"{acc * 100:.2f}%")
        if pack.get("accuracy") is not None:
            m2.metric("Accuracy saved in pkl", f"{pack['accuracy'] * 100:.2f}%")
        st.caption("Uses the same 80/20 split (random_state=42) as your training script.")

        classes = list(model.classes_)
        tick = [names.get(c, str(c)) for c in classes]
        cm = confusion_matrix(y_te, y_pred, labels=classes)
        left, right = st.columns(2)
        left.plotly_chart(px.imshow(cm, text_auto=True, x=tick, y=tick, color_continuous_scale="Blues",
                                    labels=dict(x="Predicted", y="Actual"),
                                    title="Confusion matrix"), use_container_width=True)

        if hasattr(model, "feature_importances_"):
            imp, title = model.feature_importances_, "Feature importance"
        else:
            imp, title = model.coef_[0], "Model coefficients (scaled inputs)"
        imp_df = pd.DataFrame({"Feature": [nice.get(f, f) for f in feats], "Value": imp})
        imp_df = imp_df.sort_values("Value")
        right.plotly_chart(px.bar(imp_df, x="Value", y="Feature", orientation="h", title=title),
                           use_container_width=True)

        rep = pd.DataFrame(classification_report(y_te, y_pred, output_dict=True, zero_division=0)).T
        st.subheader("Classification report")
        st.dataframe(rep.round(3), use_container_width=True)

    # ---------- Try it yourself
    with tab3:
        st.write(cfg["input_help"])
        vals, cols = {}, st.columns(3)
        for i, f in enumerate(feats):
            default = float(min(max(df[f].median(), cfg["lo"]), cfg["hi"]))
            with cols[i % 3]:
                vals[f] = st.slider(nice.get(f, f), cfg["lo"], cfg["hi"], default, cfg["step"],
                                    key=f"{cfg['pkl']}-{f}")

        user = pd.DataFrame([vals])
        proba = model.predict_proba(prep(user))[0]
        p_pos = proba[positive_index(model.classes_)]
        pred = model.predict(prep(user))[0]

        left, right = st.columns(2)
        with left:
            st.subheader(f"Prediction: {names.get(pred, str(pred))}")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p_pos * 100,
                number={"suffix": "%"},
                title={"text": f"Probability of {names.get(1, 'positive class')}"},
                gauge={"axis": {"range": [0, 100]},
                       "steps": [{"range": [0, 50], "color": "#fde2e2"},
                                 {"range": [50, 100], "color": "#d9f2e3"}]}))
            fig.update_layout(height=320, margin=dict(t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            labels = [nice.get(f, f) for f in feats]
            radar = go.Figure()
            for grp, row in df.groupby("Outcome")[feats].mean().iterrows():
                radar.add_trace(go.Scatterpolar(r=list(row.values) + [row.values[0]],
                                                theta=labels + [labels[0]], name=f"Avg {grp}",
                                                line=dict(dash="dot"), opacity=0.7))
            radar.add_trace(go.Scatterpolar(r=list(user.iloc[0].values) + [user.iloc[0].values[0]],
                                            theta=labels + [labels[0]], name="You",
                                            fill="toself", line=dict(width=3)))
            radar.update_layout(polar=dict(radialaxis=dict(range=[cfg["lo"], cfg["hi"]])),
                                title="Your profile vs group averages", height=420)
            st.plotly_chart(radar, use_container_width=True)


# ---------------------------------------------------------------- main
st.set_page_config(page_title="SDS & JDS Model Dashboard", page_icon="📊", layout="wide")
st.title("📊 SDS & JDS Model Dashboard")
choice = st.sidebar.radio("Choose a model", list(CONFIGS))
st.sidebar.caption("Predictions are statistical estimates from survey data, not guarantees.")
st.header(choice)
render(CONFIGS[choice])
