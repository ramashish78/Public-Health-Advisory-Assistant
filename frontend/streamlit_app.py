from __future__ import annotations

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import html
import json
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

from app.core.config import settings
from app.core.constants import MEDICAL_DISCLAIMER
from app.services.pipeline import AdvisoryPipeline


st.set_page_config(
    page_title="Public Health Advisory Assistant",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("PHAA_API_URL", "http://localhost:8000")
RISK_COLORS = {"LOW": "#45f0c8", "MEDIUM": "#ffb347", "HIGH": "#ff6b7a"}
CLARIFICATION_COLORS = {"LOW": "#54e0ff", "MEDIUM": "#ffb347", "HIGH": "#ff6fa6"}
SAMPLE_SCENARIOS = [
    ("Glucose Alert", "I am getting frequent urination and the blood sugar level is high from last 4 days."),
    ("Respiratory Stress", "My chest feels tight and I am short of breath with a cough."),
    ("UTI Pattern", "I have burning when I urinate and keep going to the bathroom with lower back pain."),
    ("Viral Fever", "I have fever, cough, fatigue and chills for 3 days."),
]


@st.cache_resource
def get_local_pipeline() -> AdvisoryPipeline | None:
    try:
        return AdvisoryPipeline()
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_artifact_summary() -> dict[str, str]:
    summary = {
        "accuracy": "--",
        "top_3_accuracy": "--",
        "training_rows": "--",
        "document_count": "--",
        "embedding_mode": "PENDING",
    }
    if settings.model_metrics_path.exists():
        metrics = json.loads(settings.model_metrics_path.read_text(encoding="utf-8"))
        summary["accuracy"] = f"{metrics.get('accuracy', 0) * 100:.1f}%"
        summary["top_3_accuracy"] = f"{metrics.get('top_3_accuracy', 0) * 100:.1f}%"
        summary["training_rows"] = f"{metrics.get('training_rows', 0):,}"
    if settings.retrieval_metadata_path.exists():
        metadata = json.loads(settings.retrieval_metadata_path.read_text(encoding="utf-8"))
        summary["document_count"] = str(metadata.get("document_count", "--"))
        summary["embedding_mode"] = str(metadata.get("embedding_mode", "PENDING")).upper()
    return summary


def call_api(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def analyze(text: str, top_k: int, answered_question_ids: list[str] | None = None) -> dict[str, Any]:
    payload = {
        "text": text,
        "top_k": top_k,
        "answered_question_ids": answered_question_ids or [],
    }
    try:
        return call_api("/report", payload)
    except Exception:
        pipeline = get_local_pipeline()
        if pipeline is None:
            raise RuntimeError(
                "Backend is unreachable and local pipeline artifacts are not ready. Run scripts/bootstrap_project.py."
            )
        return pipeline.report(text, top_k=top_k, answered_question_ids=answered_question_ids or [])


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root{
            --bg0:#050816;--bg1:#091326;--bg2:#140b23;--panel:rgba(10,18,35,.82);
            --line:rgba(168,190,228,.16);--text:#f7fbff;--soft:#c3d9f2;--muted:#8aa6c7;
            --cyan:#54e0ff;--teal:#45f0c8;--orange:#ffb347;--pink:#ff6fa6;
        }
        .stApp{
            color:var(--text);
            background:
                radial-gradient(circle at 10% 18%, rgba(84,224,255,.16), transparent 24%),
                radial-gradient(circle at 88% 12%, rgba(255,111,166,.16), transparent 24%),
                radial-gradient(circle at 80% 80%, rgba(255,179,71,.12), transparent 22%),
                linear-gradient(145deg, var(--bg0) 0%, var(--bg1) 52%, var(--bg2) 100%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"]{background:rgba(4,8,22,.24);backdrop-filter:blur(12px);}
        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg, rgba(9,14,30,.97), rgba(6,10,23,.97));
            border-right:1px solid var(--line);
        }
        section[data-testid="stSidebar"] *{color:#eaf4ff !important;}
        .main .block-container{max-width:1240px;padding-top:1.35rem;padding-bottom:7rem;}
        .panel,.card,.hero-card,.scenario,.risk-card,.evidence-card{
            background:linear-gradient(180deg, rgba(14,22,43,.96), rgba(8,13,26,.84));
            border:1px solid var(--line);border-radius:26px;
            box-shadow:0 28px 60px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.05);
        }
        .scroll{opacity:0;transform:translateY(28px) scale(.97);animation:enter .8s ease forwards;}
        @supports (animation-timeline:view()){.scroll{opacity:1;transform:none;animation:reveal both;animation-timeline:view();animation-range:entry 8% cover 32%;}}
        @keyframes enter{to{opacity:1;transform:none;}}
        @keyframes reveal{from{opacity:0;transform:translateY(38px) scale(.95);}to{opacity:1;transform:none;}}
        @keyframes float{0%,100%{transform:translate(-50%,-50%) rotateX(15deg) rotateY(-16deg)}50%{transform:translate(-50%,-50%) rotateX(20deg) rotateY(-6deg) translateY(-14px)}}
        @keyframes spin{from{transform:translate(-50%,-50%) rotateX(72deg) rotateZ(0deg)}to{transform:translate(-50%,-50%) rotateX(72deg) rotateZ(360deg)}}
        .hero{display:grid;grid-template-columns:1.18fr .82fr;gap:1rem;margin-bottom:1rem;}
        .hero-card{padding:2rem;position:relative;overflow:hidden;}
        .eyebrow{display:inline-block;padding:.55rem .9rem;border-radius:999px;border:1px solid rgba(84,224,255,.22);background:rgba(84,224,255,.08);color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
        .hero h1{margin:.95rem 0 .8rem;color:#fff;font-size:clamp(2.4rem,4vw,4.3rem);line-height:.95;letter-spacing:-.05em;}
        .hero p,.card p,.risk-card p,.evidence-card p{color:var(--soft);line-height:1.72;}
        .stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem;margin-top:1.1rem;}
        .stat{padding:.9rem 1rem;border-radius:18px;background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.08);transition:transform .22s ease, border-color .22s ease, box-shadow .22s ease;}
        .stat:hover,.card:hover,.scenario:hover,.risk-card:hover,.evidence-card:hover,.primary-card:hover,.section-card:hover,.other-card:hover{
            transform:translateY(-4px);
            border-color:rgba(84,224,255,.3);
            box-shadow:0 20px 40px rgba(0,0,0,.24);
        }
        .stat .label{display:block;color:var(--muted);font-size:.84rem;margin-bottom:.35rem;}
        .stat .value{display:block;color:#fff;font-size:1.35rem;font-weight:900;}
        .orb-stage{position:relative;min-height:360px;overflow:hidden;}
        .plane{position:absolute;inset:-18%;background:linear-gradient(rgba(84,224,255,.11) 1px, transparent 1px), linear-gradient(90deg, rgba(84,224,255,.11) 1px, transparent 1px);background-size:42px 42px;transform:rotateX(76deg) translateY(20px);opacity:.3;}
        .orb{position:absolute;left:50%;top:50%;width:185px;height:185px;border-radius:50%;background:radial-gradient(circle at 32% 28%, rgba(255,255,255,.96), rgba(84,224,255,.35) 24%, rgba(84,224,255,.08) 58%, rgba(255,255,255,.02) 100%);box-shadow:0 0 80px rgba(84,224,255,.24), inset 0 0 28px rgba(255,255,255,.2);animation:float 8s ease-in-out infinite;}
        .ring{position:absolute;left:50%;top:50%;border-radius:50%;border:1px solid rgba(84,224,255,.44);animation:spin 15s linear infinite;}
        .r1{width:270px;height:270px;}.r2{width:330px;height:330px;border-color:rgba(255,179,71,.42);animation-duration:20s;}.r3{width:390px;height:390px;border-color:rgba(255,111,166,.24);animation-duration:26s;}
        .node{position:absolute;padding:.68rem .9rem;border-radius:16px;background:rgba(10,18,35,.9);border:1px solid rgba(255,255,255,.08);font-weight:800;color:#fff;font-size:.86rem;}
        .node small{display:block;color:var(--muted);font-weight:700;margin-top:.18rem;}
        .n1{left:8%;top:18%;}.n2{right:10%;top:24%;}.n3{left:14%;bottom:17%;}.n4{right:14%;bottom:15%;}
        .section-kicker{color:var(--cyan);font-size:.8rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.4rem;}
        .section-title{font-size:1.6rem;font-weight:900;color:#fff;letter-spacing:-.03em;margin:0 0 .35rem;}
        .scenario{padding:1.1rem;min-height:292px;display:flex;flex-direction:column;justify-content:space-between;}
        .scenario h4{margin:0;color:#fff;font-size:1.04rem;}
        .scenario p{color:var(--muted);min-height:48px;margin:.9rem 0 1rem;}
        .scenario code{display:block;white-space:pre-wrap;padding:.82rem;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);color:#e3f5ff;font-size:.86rem;min-height:108px;}
        .result-shell{display:grid;grid-template-columns:1.18fr .82fr;gap:1rem;margin-top:1rem;}
        .primary-card{padding:1.35rem 1.4rem;position:relative;overflow:hidden;}
        .primary-card::after{
            content:"";
            position:absolute;
            right:-70px;top:-70px;width:220px;height:220px;border-radius:50%;
            background:radial-gradient(circle, rgba(84,224,255,.18), transparent 70%);
            pointer-events:none;
        }
        .badge{display:inline-flex;align-items:center;gap:.4rem;padding:.52rem .86rem;border-radius:999px;background:rgba(84,224,255,.08);border:1px solid rgba(84,224,255,.2);color:var(--cyan);font-size:.75rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
        .disease-row{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-top:.95rem;}
        .disease-name{color:#fff;font-size:2.15rem;font-weight:900;line-height:1.02;letter-spacing:-.04em;margin:0 0 .45rem;}
        .summary-copy{color:var(--soft);line-height:1.76;margin:0;}
        .confidence-pill{padding:.7rem 1rem;border-radius:18px;background:linear-gradient(135deg, rgba(84,224,255,.18), rgba(255,179,71,.15));border:1px solid rgba(255,255,255,.09);min-width:150px;text-align:center;}
        .confidence-pill .label{display:block;color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.08em;}
        .confidence-pill .value{display:block;color:#fff;font-size:1.55rem;font-weight:900;margin-top:.2rem;}
        .focus-chips{display:flex;gap:.65rem;flex-wrap:wrap;margin-top:1rem;}
        .focus-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:1rem;}
        .section-card,.other-card{padding:1.15rem 1.2rem;border-radius:22px;background:linear-gradient(180deg, rgba(14,22,43,.92), rgba(8,13,26,.8));border:1px solid rgba(168,190,228,.14);height:100%;}
        .section-card h4,.other-card h4{margin:0 0 .55rem;color:#fff;font-size:1rem;}
        .section-card p,.other-card p{color:var(--soft);line-height:1.72;margin:.2rem 0;}
        .clarify-shell{padding:1.2rem 1.25rem;margin:1rem 0 1.05rem;}
        .clarify-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-bottom:.95rem;}
        .clarify-head h3{margin:0;color:#fff;font-size:1.28rem;letter-spacing:-.03em;}
        .clarify-head p{margin:.35rem 0 0;color:var(--soft);line-height:1.72;}
        .clarify-note{margin-top:.7rem;color:var(--muted);font-size:.92rem;}
        .clarify-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;}
        .clarify-card{padding:1.05rem 1.1rem;border-radius:22px;background:linear-gradient(180deg, rgba(14,22,43,.92), rgba(8,13,26,.8));border:1px solid rgba(168,190,228,.14);min-height:180px;}
        .clarify-card h4{margin:.15rem 0 .55rem;color:#fff;font-size:1.02rem;line-height:1.45;}
        .clarify-card p{margin:0;color:var(--soft);line-height:1.68;}
        .report-gap{height:1rem;}
        .stack-list{margin:.25rem 0 0;padding-left:1rem;color:var(--soft);line-height:1.72;}
        .stack-list li{margin-bottom:.36rem;}
        .subtext{color:var(--muted);font-size:.9rem;}
        .other-shell{display:grid;gap:.85rem;}
        .top-grid{display:grid;grid-template-columns:1.16fr .84fr;gap:1rem;}
        .card,.risk-card,.evidence-card{padding:1.2rem 1.25rem;}
        .mini{display:inline-block;color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.65rem;}
        .query{color:#fff;font-size:1.24rem;font-weight:900;line-height:1.45;margin-bottom:.55rem;}
        .chips,.meta{display:flex;gap:.65rem;flex-wrap:wrap;margin-top:.8rem;}
        .chip{padding:.58rem .88rem;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#edf6ff;font-weight:800;font-size:.83rem;}
        .risk-level{display:inline-flex;padding:.66rem 1rem;border-radius:999px;color:#08101f;font-size:.78rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.8rem;}
        .bullet{margin:.4rem 0 0;padding-left:1.1rem;color:var(--soft);line-height:1.7;}
        .bullet li{margin-bottom:.4rem;}
        .pred-grid,.evi-grid{display:grid;gap:.85rem;}
        .pred-top{border-color:rgba(84,224,255,.42);}
        .pred-head{display:flex;align-items:center;justify-content:space-between;gap:.7rem;margin-bottom:.4rem;}
        .rank{display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;border-radius:50%;background:linear-gradient(135deg,var(--cyan),var(--orange));color:#07101d;font-weight:900;}
        .prob{color:#fff;font-size:1.04rem;font-weight:900;}
        .tag{padding:.42rem .7rem;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:var(--muted);font-size:.78rem;font-weight:700;}
        .stTabs [data-baseweb="tab-list"]{gap:.7rem;margin-bottom:1rem;}
        .stTabs [data-baseweb="tab"]{border:1px solid var(--line);background:rgba(10,18,35,.82);border-radius:999px;padding:.7rem 1.1rem;color:#dcecff;font-weight:700;}
        .stTabs [aria-selected="true"]{background:linear-gradient(135deg, rgba(84,224,255,.96), rgba(255,179,71,.96));color:#07111d !important;border-color:transparent !important;}
        .stTabs [data-baseweb="tab-highlight"]{display:none !important;}
        .stTabs [data-baseweb="tab"] button,.stTabs [data-baseweb="tab"]:focus,.stTabs [data-baseweb="tab"]:focus-visible{outline:none !important;box-shadow:none !important;}
        .stTabs button:focus,.stTabs button:focus-visible{outline:none !important;box-shadow:none !important;}
        .stButton>button,.stDownloadButton>button{width:100%;min-height:3rem;border-radius:18px;border:1px solid rgba(255,255,255,.10);background:linear-gradient(135deg, rgba(14,22,43,.98), rgba(9,15,28,.92));color:#f6fbff;box-shadow:0 18px 34px rgba(0,0,0,.28);font-weight:800;white-space:normal;transition:transform .22s ease, border-color .22s ease;}
        .stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-4px);border-color:rgba(84,224,255,.34);}
        div[data-testid="stChatInput"]{border-radius:24px;border:1px solid rgba(255,255,255,.08);background:linear-gradient(135deg, rgba(9,14,28,.98), rgba(6,11,22,.96));box-shadow:0 28px 56px rgba(0,0,0,.34);}
        div[data-testid="stChatInput"] textarea,div[data-testid="stChatInput"] input{color:#f7fbff !important;caret-color:var(--cyan) !important;}
        div[data-testid="stChatInput"] textarea::placeholder,div[data-testid="stChatInput"] input::placeholder{color:#90aac6 !important;}
        div[data-testid="stChatInput"] button{border-radius:50%;background:linear-gradient(135deg, var(--cyan), var(--orange));}
        [data-testid="stExpander"]{border:1px solid var(--line);border-radius:22px;background:rgba(10,17,33,.74);}
        @media (max-width:960px){.hero,.top-grid,.stats,.result-shell,.focus-grid,.clarify-grid{grid-template-columns:1fr;}.orb-stage{min-height:320px;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, str | None]:
    choice = None
    with st.sidebar:
        st.markdown("## Control Deck")
        st.caption("Launch curated demos and tune how many possible conditions appear.")
        top_k = st.slider("Top conditions", min_value=1, max_value=5, value=3)
        st.markdown("### Quick launch")
        for index, (label, prompt) in enumerate(SAMPLE_SCENARIOS):
            if st.button(label, key=f"sb_{index}", use_container_width=True):
                choice = prompt
        st.markdown("### Safety notice")
        st.caption(MEDICAL_DISCLAIMER)
    return top_k, choice


def render_hero(summary: dict[str, str], report_count: int) -> None:
    st.markdown(
        f"""
        <div class="hero scroll">
            <div class="hero-card">
                <span class="eyebrow">Local AI + NLP + RAG + Safety Rules</span>
                <h1>Public Health Advisory Assistant</h1>
                <p>
                    A high-contrast clinical cockpit with richer motion, layered glass cards, hover depth,
                    and safer symptom fusion. This build now treats frequent urination plus high blood sugar
                    as a metabolic concern rather than defaulting to a urinary infection.
                </p>
                <div class="stats">
                    <div class="stat"><span class="label">ML accuracy</span><span class="value">{summary['accuracy']}</span></div>
                    <div class="stat"><span class="label">Top-3 capture</span><span class="value">{summary['top_3_accuracy']}</span></div>
                    <div class="stat"><span class="label">Session analyses</span><span class="value">{report_count}</span></div>
                    <div class="stat"><span class="label">Training rows</span><span class="value">{summary['training_rows']}</span></div>
                    <div class="stat"><span class="label">RAG documents</span><span class="value">{summary['document_count']}</span></div>
                    <div class="stat"><span class="label">Embedding mode</span><span class="value">{summary['embedding_mode']}</span></div>
                </div>
            </div>
            <div class="hero-card orb-stage">
                <div class="plane"></div>
                <div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div><div class="orb"></div>
                <div class="node n1">Symptom NLP<small>entity extraction</small></div>
                <div class="node n2">Classifier<small>probability fusion</small></div>
                <div class="node n3">RAG Evidence<small>FAISS retrieval</small></div>
                <div class="node n4">Risk Engine<small>rule-based safety</small></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scenarios() -> str | None:
    st.markdown(
        """
        <div class="scroll">
            <div class="section-kicker">Interactive Launchpad</div>
            <div class="section-title">Try live scenarios</div>
            <p style="color:#c3d9f2;margin-top:0;">Each card runs the same end-to-end advisory flow as manual chat input.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = None
    cols = st.columns(len(SAMPLE_SCENARIOS))
    for index, (label, prompt) in enumerate(SAMPLE_SCENARIOS):
        with cols[index]:
            st.markdown(
                f"""
                <div class="scenario scroll">
                    <h4>{html.escape(label)}</h4>
                    <p>Click to run this scenario instantly.</p>
                    <code>{html.escape(prompt)}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Run scenario", key=f"sc_{index}", use_container_width=True):
                selected = prompt
    return selected


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="panel scroll" style="padding:1.6rem;text-align:center;">
            <div class="section-kicker">Ready For Input</div>
            <div class="section-title">Describe a symptom pattern to open the advisory dashboard</div>
            <p>The assistant will extract symptoms, score condition patterns, retrieve supporting medical context, and produce a conservative risk-aware summary.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_primary_condition(report: dict[str, Any]) -> dict[str, Any]:
    primary = report.get("primary_condition")
    if primary:
        return primary

    top_prediction = report.get("predictions", [{}])[0]
    return {
        "disease": top_prediction.get("disease", "Unclear pattern"),
        "confidence": top_prediction.get("probability", 0.0),
        "description": top_prediction.get("description", ""),
        "matched_symptoms": report.get("symptoms", []),
        "hallmark_symptoms": [],
        "related_symptoms_to_watch": [],
        "care_priorities": report.get("advice", {}).get("care_steps", [])[:3],
        "escalation": report.get("advice", {}).get("when_to_escalate", ""),
        "focused_evidence": [],
    }


def build_follow_up_text(base_text: str, details: list[str]) -> str:
    cleaned_details = []
    for detail in details:
        detail = detail.strip()
        if detail and detail not in cleaned_details:
            cleaned_details.append(detail)
    if not cleaned_details:
        return base_text
    suffix = " ".join(cleaned_details)
    return f"{base_text.strip()} Additional detail: {suffix}"


def render_clarification_panel(report: dict[str, Any], top_k: int, interactive: bool = True) -> None:
    clarification = report.get("clarification", {})
    if not clarification.get("should_clarify"):
        return

    priority = clarification.get("priority", "LOW")
    priority_color = CLARIFICATION_COLORS.get(priority, "#54e0ff")
    questions = clarification.get("questions", [])

    st.markdown(
        f"""
        <div class="clarify-shell panel scroll">
            <div class="clarify-head">
                <div>
                    <span class="badge">Clarification recommended</span>
                    <h3>Answer one or more focused follow-up questions</h3>
                    <p>{html.escape(clarification.get('summary', ''))}</p>
                </div>
                <div class="confidence-pill" style="min-width:180px;">
                    <span class="label">Priority</span>
                    <span class="value" style="color:{priority_color};">{html.escape(priority)}</span>
                </div>
            </div>
            <p class="clarify-note">{html.escape(clarification.get('rationale', ''))}</p>
            <p class="clarify-note">{html.escape(clarification.get('suggested_reply', ''))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not questions:
        return

    if not interactive:
        columns = st.columns(len(questions), gap="large")
        for index, question in enumerate(questions):
            with columns[index]:
                options = "".join(
                    f'<span class="chip">{html.escape(option["label"])}</span>'
                    for option in question.get("options", [])
                )
                st.markdown(
                    f"""
                    <div class="clarify-card scroll">
                        <span class="mini">Follow-up {index + 1}</span>
                        <h4>{html.escape(question['question'])}</h4>
                        <p>{html.escape(question['why'])}</p>
                        <div class="chips" style="margin-top:1rem;">{options}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    with st.form(key=f"clarify_form_{report['report_id']}"):
        columns = st.columns(len(questions), gap="large")
        selections: list[tuple[dict[str, Any], str]] = []
        for index, question in enumerate(questions):
            with columns[index]:
                st.markdown(
                    f"""
                    <div class="clarify-card">
                        <span class="mini">Follow-up {index + 1}</span>
                        <h4>{html.escape(question['question'])}</h4>
                        <p>{html.escape(question['why'])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                labels = ["Skip this for now", *[option["label"] for option in question.get("options", [])]]
                choice = st.selectbox(
                    f"Choose an answer for follow-up {index + 1}",
                    labels,
                    key=f"clarify_{report['report_id']}_{question['id']}",
                    label_visibility="collapsed",
                )
                selections.append((question, choice))
        submitted = st.form_submit_button("Reanalyze with follow-up details", use_container_width=True)

    if submitted:
        selected_details: list[str] = []
        handled_question_ids = [question["id"] for question, _ in selections]
        for question, choice in selections:
            if choice == "Skip this for now":
                continue
            option = next(
                (item for item in question.get("options", []) if item["label"] == choice),
                None,
            )
            if option:
                selected_details.append(option["append_text"])
        if not selected_details:
            st.toast("Select at least one follow-up answer first.")
            return
        run_analysis(
            build_follow_up_text(report["input_text"], selected_details),
            top_k,
            answered_question_ids=handled_question_ids,
        )


def render_primary_focus(report: dict[str, Any]) -> None:
    primary = get_primary_condition(report)
    clarification = report.get("clarification", {})
    is_uncertain = primary.get("is_uncertain", False)
    level = report["risk"]["level"]
    color = RISK_COLORS.get(level, "#54e0ff")
    chips = "".join(
        f'<span class="chip">{html.escape(item)}</span>' for item in primary.get("matched_symptoms", [])
    ) or '<span class="chip">No hallmark symptom match extracted</span>'
    watch_list = "".join(
        f"<li>{html.escape(item)}</li>" for item in primary.get("related_symptoms_to_watch", [])
    ) or "<li>No additional hallmark symptoms were highlighted for this condition profile.</li>"
    care_priorities = "".join(
        f"<li>{html.escape(item)}</li>" for item in primary.get("care_priorities", [])
    ) or "<li>No disease-specific care priorities were available.</li>"
    focused_evidence = primary.get("focused_evidence", [])
    evidence_text = (
        html.escape(focused_evidence[0]["snippet"])
        if focused_evidence
        else html.escape(report["advice"]["summary"])
    )
    evidence_source = (
        html.escape(focused_evidence[0]["source"])
        if focused_evidence
        else "Advisory synthesis"
    )
    reasons = "".join(f"<li>{html.escape(item)}</li>" for item in report["risk"]["reasons"]) or (
        "<li>No extra risk rationale was generated.</li>"
    )
    notes = "".join(
        f"<li>{html.escape(note)}</li>" for note in report.get("decision_support", {}).get("rule_notes", [])
    ) or "<li>No extra reranking rule fired for this input.</li>"
    badge_text = "Needs Review" if is_uncertain else "Primary match"
    disease_title = "No exact disease confidently matched" if is_uncertain else primary["disease"]
    confidence_label = "Top model score" if is_uncertain else "Confidence"
    meta_chips = [
        f'<span class="chip">Input focus: {html.escape(report["input_text"])}</span>',
        f'<span class="chip">Report ID: {html.escape(report["report_id"])}</span>',
    ]
    if clarification.get("should_clarify"):
        meta_chips.append(
            f'<span class="chip">Follow-up questions: {clarification.get("question_count", 0)}</span>'
        )
    meta_html = "".join(meta_chips)

    st.markdown(
        f"""
        <div class="result-shell scroll">
            <div class="primary-card panel">
                <span class="badge">{html.escape(badge_text)}</span>
                <div class="disease-row">
                    <div>
                        <div class="disease-name">{html.escape(disease_title)}</div>
                        <p class="summary-copy">{html.escape(primary.get('description', ''))}</p>
                    </div>
                    <div class="confidence-pill">
                        <span class="label">{html.escape(confidence_label)}</span>
                        <span class="value">{primary.get('confidence', 0) * 100:.1f}%</span>
                    </div>
                </div>
                <p style="margin-top:.95rem;">{html.escape(report['advice']['explanation'])}</p>
                <div class="focus-chips">{chips}</div>
                <div class="meta">{meta_html}</div>
            </div>
            <div class="risk-card">
                <span class="mini">Highlighted action</span>
                <div class="risk-level" style="background:{color};">{html.escape(level)} risk</div>
                <p>{html.escape(report['risk']['recommended_action'])}</p>
                <p class="subtext">Top evidence: {evidence_source}</p>
                <p>{evidence_text}</p>
                <div class="meta">
                    <span class="chip">Matched phrases: {html.escape(', '.join(report.get('matched_phrases', [])) or 'No direct alias matched')}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="focus-grid scroll">
            <div class="section-card">
                <span class="mini">Related information</span>
                <h4>What this condition pattern means</h4>
                <p>{html.escape(report['advice']['summary'])}</p>
                <p>{evidence_text}</p>
            </div>
            <div class="section-card">
                <span class="mini">Symptoms to focus on</span>
                <h4>Matched and related symptoms</h4>
                <p><strong style="color:#fff;">Matched hallmark symptoms:</strong></p>
                <div class="focus-chips">{chips}</div>
                <p style="margin-top:.85rem;"><strong style="color:#fff;">Other related symptoms to watch:</strong></p>
                <ul class="stack-list">{watch_list}</ul>
            </div>
            <div class="section-card">
                <span class="mini">Immediate priorities</span>
                <h4>What to do next</h4>
                <ul class="stack-list">{care_priorities}</ul>
                <p><strong style="color:#fff;">Escalate if:</strong> {html.escape(primary.get('escalation', report['advice']['when_to_escalate']))}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="report-gap"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            f"""
            <div class="section-card scroll">
                <span class="mini">Why it was highlighted</span>
                <h4>Clinical reasoning</h4>
                <ul class="stack-list">{reasons}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="section-card scroll">
                <span class="mini">Decision fusion</span>
                <h4>Rule-based adjustments</h4>
                <ul class="stack-list">{notes}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_chart(predictions: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(predictions)
    if frame.empty:
        return
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=14, cornerRadiusBottomRight=14)
        .encode(
            x=alt.X("probability:Q", title="Confidence", axis=alt.Axis(format="%")),
            y=alt.Y("disease:N", sort="-x", title=None),
            color=alt.Color("disease:N", legend=None, scale=alt.Scale(range=["#54e0ff", "#ffb347", "#ff6fa6", "#45f0c8", "#96a0ff"])),
            tooltip=[
                alt.Tooltip("disease:N", title="Condition"),
                alt.Tooltip("probability:Q", title="Confidence", format=".1%"),
                alt.Tooltip("description:N", title="Summary"),
            ],
        )
        .properties(height=300)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#dcecff", titleColor="#9bc7ff", gridColor="rgba(159,180,212,.16)")
    )
    st.altair_chart(chart, use_container_width=True)


def render_other_possibilities(report: dict[str, Any]) -> None:
    predictions = report.get("predictions", [])
    primary = predictions[0] if predictions else None
    alternatives = predictions[1:]
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        if primary:
            st.markdown(
                f"""
                <div class="section-card scroll">
                    <span class="mini">Primary result</span>
                    <h4>{html.escape('No exact disease confidently matched' if primary.get('disease') == 'Unclear Pattern / Needs Clinical Evaluation' else primary['disease'])}</h4>
                    <p>{html.escape(primary.get('description', ''))}</p>
                    <p><strong style="color:#fff;">{'Top model score' if primary.get('disease') == 'Unclear Pattern / Needs Clinical Evaluation' else 'Confidence'}:</strong> {primary['probability'] * 100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if alternatives:
            for index, item in enumerate(alternatives, start=2):
                st.markdown(
                    f"""
                    <div class="other-card scroll">
                        <div class="pred-head">
                            <div style="display:flex;align-items:center;gap:.75rem;">
                                <span class="rank">{index}</span>
                                <div>
                                    <h4>{html.escape(item['disease'])}</h4>
                                    <p>{html.escape(item.get('description', ''))}</p>
                                </div>
                            </div>
                            <div class="prob">{item['probability'] * 100:.1f}%</div>
                        </div>
                    </div>
                    """.strip(),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="section-card scroll">
                    <span class="mini">Other possibilities</span>
                    <h4>No secondary condition patterns were returned.</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            """
            <div class="section-card scroll">
                <span class="mini">Ranking overview</span>
                <h4>How the alternatives compare</h4>
                <p>The chart is kept below the primary result so the leading condition stays front and center.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_prediction_chart(predictions)


def render_guidance(report: dict[str, Any]) -> None:
    primary = get_primary_condition(report)
    steps = "".join(f"<li>{html.escape(step)}</li>" for step in report["advice"]["care_steps"])
    snippets = "".join(
        f"<li><strong>{html.escape(item['title'])}</strong>: {html.escape(item['snippet'])}</li>"
        for item in report["advice"].get("evidence_snippets", [])
    ) or "<li>No short evidence snippets were generated.</li>"
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            f"""
            <div class="card scroll">
                <span class="mini">Care guidance</span>
                <div class="query" style="font-size:1.06rem;">Action plan for {html.escape(primary['disease'])}</div>
                <ul class="bullet">{steps}</ul>
                <p><strong style="color:#fff;">Escalate if:</strong> {html.escape(primary.get('escalation', report['advice']['when_to_escalate']))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"""
            <div class="card scroll">
                <span class="mini">Evidence summary</span>
                <div class="query" style="font-size:1.06rem;">Retrieved support</div>
                <ul class="bullet">{snippets}</ul>
                <p>{html.escape(report['advice']['disclaimer'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence(report: dict[str, Any]) -> None:
    primary = get_primary_condition(report)
    focused_titles = {item["title"] for item in primary.get("focused_evidence", [])}
    cards = []
    for item in report.get("retrieval_results", []):
        emphasis = " evidence-card scroll" if item["title"] not in focused_titles else " evidence-card pred-top scroll"
        tags = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in item.get("tags", []))
        cards.append(
            f"""
            <div class="{emphasis}">
                <span class="mini">Retrieved context</span>
                <div class="query" style="font-size:1.05rem;">{html.escape(item['title'])}</div>
                <div class="meta">
                    <span class="tag">{html.escape(item['source'])}</span>
                    <span class="tag">Disease: {html.escape(item['disease'])}</span>
                    <span class="tag">Similarity: {item['score']:.4f}</span>
                    {tags}
                </div>
                <p>{html.escape(item['content'])}</p>
                <p><strong style="color:#fff;">Precautions:</strong> {html.escape(', '.join(item.get('precautions', [])))}</p>
                <p><strong style="color:#fff;">Escalation:</strong> {html.escape(item.get('escalation', ''))}</p>
            </div>
            """
        )
    st.markdown("".join(cards) or "<p>No retrieval results available.</p>", unsafe_allow_html=True)


def render_technical(report: dict[str, Any]) -> None:
    clarification = report.get("clarification", {})
    c1, c2 = st.columns([0.8, 1.2], gap="large")
    with c1:
        st.markdown(
            f"""
            <div class="card scroll">
                <span class="mini">Technical snapshot</span>
                <div class="query" style="font-size:1.06rem;">Extraction and report metadata</div>
                <div class="meta">
                    <span class="chip">Symptoms: {len(report.get('symptoms', []))}</span>
                    <span class="chip">Matched: {len(report.get('matched_phrases', []))}</span>
                    <span class="chip">RAG hits: {len(report.get('retrieval_results', []))}</span>
                    <span class="chip">Clarify: {'Yes' if clarification.get('should_clarify') else 'No'}</span>
                </div>
                <p>Created at: {html.escape(report['created_at'])}</p>
                <p>Normalized text: {html.escape(report['normalized_text'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name=f"{report['report_id']}.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.code(json.dumps(report, indent=2), language="json")


def render_report(report: dict[str, Any], label: str, top_k: int, interactive: bool = True) -> None:
    st.markdown(
        f"""
        <div class="scroll">
            <div class="section-kicker">{html.escape(label)}</div>
            <div class="section-title">Advisory dashboard</div>
            <p style="color:#c3d9f2;margin-top:0;">The primary condition pattern is highlighted first, with evidence and alternatives organized underneath it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_primary_focus(report)
    render_clarification_panel(report, top_k=top_k, interactive=interactive)
    st.markdown('<div class="report-gap"></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Care Plan", "Supporting Evidence", "Other Possibilities", "Technical"])
    with tabs[0]:
        render_guidance(report)
    with tabs[1]:
        render_evidence(report)
    with tabs[2]:
        render_other_possibilities(report)
    with tabs[3]:
        render_technical(report)


def run_analysis(text: str, top_k: int, answered_question_ids: list[str] | None = None) -> None:
    with st.spinner("Analyzing symptoms, retrieving evidence, and building the advisory dashboard..."):
        report = analyze(text, top_k, answered_question_ids=answered_question_ids)
    st.session_state.reports.insert(0, report)
    st.toast("Analysis ready")
    st.rerun()


def main() -> None:
    inject_styles()
    top_k, sidebar_prompt = render_sidebar()
    if "reports" not in st.session_state:
        st.session_state.reports = []

    render_hero(load_artifact_summary(), len(st.session_state.reports))
    main_prompt = render_scenarios()
    prompt = sidebar_prompt or main_prompt
    if prompt:
        run_analysis(prompt, top_k)

    if st.session_state.reports:
        render_report(st.session_state.reports[0], "Latest analysis", top_k=top_k, interactive=True)
        if len(st.session_state.reports) > 1:
            st.markdown('<div class="section-kicker">Session memory</div><div class="section-title">Previous runs</div>', unsafe_allow_html=True)
            for index, report in enumerate(st.session_state.reports[1:], start=2):
                preview = report["input_text"][:90] + ("..." if len(report["input_text"]) > 90 else "")
                with st.expander(f"Analysis {index}: {preview}", expanded=False):
                    render_report(report, f"History item {index}", top_k=top_k, interactive=False)
    else:
        render_empty_state()

    user_text = st.chat_input("Describe your symptoms in natural language...")
    if user_text:
        run_analysis(user_text, top_k)


if __name__ == "__main__":
    main()
