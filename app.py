"""
Streamlit UI for the multi-agent research system.

Run with:
    streamlit run app.py

Place this file next to agents.py, tools.py and pipeline.py.
Requires Streamlit 1.39+ (uses st.container(key=...) for styling).
"""

import html
import math
import random
import time
import traceback
from functools import lru_cache
from urllib.parse import quote

import streamlit as st

from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    writer_chain,
)

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Desk",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Palette
#   ink       #10233F  hero band, headings
#   fog       #EEF1F5  page background
#   cobalt    #5B6BFF  finished steps
#   marigold  #FFB627  live step, primary action, critic annotation rule
#   slate     #5A667A  secondary text
#   coral     #E5533D  errors
# Type
#   Bricolage Grotesque for interface + headings, Source Serif 4 for the report body.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
:root{--ink:#10233F;--fog:#EEF1F5;--cobalt:#5B6BFF;--marigold:#FFB627;--slate:#5A667A;--line:#D5DBE5;--coral:#E5533D;--sans:'Bricolage Grotesque',system-ui,-apple-system,'Segoe UI',sans-serif;--serif:'Source Serif 4',Georgia,'Times New Roman',serif;}
.stApp{background:#080F22;color:#E6ECF8;color-scheme:light;}
.stApp,.stApp p,.stApp label,.stApp button,.stApp input,.stApp textarea,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:var(--sans);}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none !important;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:100%;padding:1.8rem 5vw 4rem 5vw;}
/* ---------- hero band ---------- */
.st-key-hero{background:transparent;border:none;box-shadow:none;padding:1.2rem 0 .5rem;}
.hero-title{color:#fff;font-family:var(--sans);font-weight:800;font-size:clamp(2rem,4.6vw,3.3rem);line-height:1.04;letter-spacing:-0.03em;margin:0 0 .7rem;padding:0;}
.hero-sub{color:#DCE6F7;font-size:1.12rem;line-height:1.5;max-width:38rem;margin:0 0 2.2rem;}
/* ---------- pipeline tracker ---------- */
.tracker{display:flex;margin:0 0 1.9rem;}
.stn{flex:1;position:relative;display:flex;flex-direction:column;gap:.15rem;padding-right:1.1rem;}
.stn::before{content:"";position:absolute;top:17px;left:48px;right:10px;height:2px;background:#2C4470;border-radius:2px;}
.stn:last-child::before{display:none;}
.stn.done::before{background:var(--cobalt);}
.dot{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.95rem;margin-bottom:.6rem;border:2px solid #34507D;color:#8FA3C4;}
.stn.done .dot{background:var(--cobalt);border-color:var(--cobalt);color:#fff;}
.stn.active .dot{background:var(--marigold);border-color:var(--marigold);color:var(--ink);animation:pulse 1.6s ease-out infinite;}
.stn.error .dot{background:var(--coral);border-color:var(--coral);color:#fff;}
.stn b{color:#fff;font-weight:700;font-size:1.05rem;}
.stn span{color:#D3DDF0;font-size:.9rem;line-height:1.35;}
.stn em{color:var(--marigold);font-style:normal;font-size:.88rem;font-weight:700;}
.stn.idle b{color:#fff;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,182,39,.55);}100%{box-shadow:0 0 0 14px rgba(255,182,39,0);}}
@media (prefers-reduced-motion:reduce){.stn.active .dot{animation:none;}}
@media (max-width:700px){.st-key-hero{padding:1rem 0;}.tracker{flex-direction:column;gap:1rem;}.stn{flex-direction:row;flex-wrap:wrap;align-items:center;column-gap:.8rem;}.stn::before{display:none;}.dot{margin-bottom:0;}}
.run-note{color:var(--marigold);font-size:1rem;margin:-.6rem 0 1.3rem;font-weight:600;}
/* ---------- input + actions (inside hero) ---------- */
.st-key-hero [data-testid="stForm"]{border:none;padding:0;background:transparent;max-width:64rem;}
.st-key-hero div[data-baseweb="input"],.st-key-hero div[data-baseweb="base-input"]{background:rgba(255,255,255,.08) !important;border-radius:12px !important;border:1px solid #6F8BC4 !important;}
.st-key-hero input{color:#fff !important;-webkit-text-fill-color:#fff !important;caret-color:#fff;font-size:1.15rem;padding:1rem 1.1rem;}
.st-key-hero input::placeholder{color:#B7C3D9 !important;-webkit-text-fill-color:#B7C3D9 !important;opacity:1;}
.st-key-hero [data-testid="stFormSubmitButton"] button{width:100%;height:3.35rem;background:linear-gradient(90deg,#7C4DFF 0%,#3D7BFF 100%);color:#fff !important;border:none;border-radius:12px;font-weight:700;font-size:1.02rem;box-shadow:0 8px 26px -8px rgba(124,77,255,.75);}
.st-key-hero [data-testid="stFormSubmitButton"] button *{color:#fff !important;}
.st-key-hero [data-testid="stFormSubmitButton"] button:hover{background:linear-gradient(90deg,#8F65FF 0%,#5A91FF 100%);color:#fff;}
.try-line{color:#DCE6F7;font-size:.92rem;margin:1.1rem 0 .4rem;}
.st-key-chips{max-width:64rem;}
.st-key-chips button{background:rgba(255,255,255,.06);border:1px solid #6F8BC4;color:#fff !important;border-radius:999px;padding:.25rem 1rem;font-size:.92rem;}
.st-key-chips button *{color:#fff !important;}
.st-key-chips button:hover{border-color:var(--marigold);background:rgba(255,182,39,.12);}
.st-key-chips button:focus-visible,.st-key-hero button:focus-visible{outline:2px solid var(--marigold);outline-offset:2px;}
/* ---------- results ---------- */
.result-meta{color:#A9B8D4;font-size:1rem;margin:2.2rem 0 1.1rem;}
.result-meta b{color:#fff;}
.st-key-report{background:#fff;border:1px solid var(--line);border-radius:16px;padding:2.4rem 2.7rem 2rem;}
.st-key-report,.st-key-report *{color:var(--ink);}
.st-key-report a,.st-key-report a *{color:#3346E0;}
.st-key-report code{background:#EEF1F7;color:#1B2A4A;padding:.1rem .35rem;border-radius:4px;}
.st-key-report blockquote,.st-key-report blockquote *{color:#3A4A66;}
.st-key-report p,.st-key-report li{font-family:var(--serif);font-size:1.07rem;line-height:1.78;}
.st-key-report h1,.st-key-report h2,.st-key-report h3,.st-key-report h4{font-family:var(--sans);color:var(--ink);letter-spacing:-0.015em;font-weight:800;}
.st-key-report h1{font-size:1.9rem;line-height:1.15;}
.st-key-report h2{font-size:1.38rem;margin-top:1.8rem;}
.st-key-report h3{font-size:1.12rem;}
.st-key-report a{color:#3346E0;}
.st-key-report [data-testid="stDownloadButton"] button{background:var(--ink);color:#fff;border:none;border-radius:10px;font-weight:600;padding:.5rem 1.1rem;margin-top:1.2rem;}
.st-key-report [data-testid="stDownloadButton"] button,.st-key-report [data-testid="stDownloadButton"] button *{color:#fff;}
.st-key-report p,.st-key-report li{max-width:46rem;}
.st-key-report [data-testid="stDownloadButton"] button:hover{background:#1E3A66;color:#fff;}
.st-key-notes{border-left:3px solid var(--marigold);padding:.1rem 0 .1rem 1.5rem;}
.notes-title{font-family:var(--sans);font-weight:800;font-size:1.3rem;letter-spacing:-0.015em;color:#fff;margin:0 0 .8rem;}
.st-key-notes,.st-key-notes *{color:#E3EBF8;}
.st-key-notes h1,.st-key-notes h2,.st-key-notes h3,.st-key-notes h4,.st-key-notes strong,.st-key-notes b{color:#fff;}
.st-key-notes a,.st-key-notes a *{color:#9FC0FF;}
.st-key-notes p,.st-key-notes li{font-size:1rem;line-height:1.65;}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line) !important;border-radius:12px;}
[data-testid="stExpander"],[data-testid="stExpander"] *{color:var(--ink);}
[data-testid="stExpander"] a,[data-testid="stExpander"] a *{color:#3346E0;}
.st-key-sources{margin-top:1.6rem;}
[data-testid="stAlert"]{background:#fff;border-radius:12px;}
[data-testid="stAlert"],[data-testid="stAlert"] *{color:var(--ink);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Background: neural-network constellation + glow orbs + faint grid.
# Generated in code, so there is no image file to host or download.
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def build_network_svg(width: int = 1600, height: int = 900, count: int = 75, seed: int = 11) -> str:
    rnd = random.Random(seed)
    points = [(rnd.uniform(0, width), rnd.uniform(0, height)) for _ in range(count)]
    reach = 220

    lines = []
    for i, (x1, y1) in enumerate(points):
        for x2, y2 in points[i + 1:]:
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < reach:
                opacity = 0.34 * (1 - dist / reach)
                lines.append(
                    f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' "
                    f"stroke='#8FA8FF' stroke-opacity='{opacity:.2f}'/>"
                )

    nodes = []
    for i, (x, y) in enumerate(points):
        if i % 6 == 0:  # bright, softly pulsing nodes
            delay = rnd.uniform(0, 4)
            nodes.append(
                f"<g class='t' style='animation-delay:{delay:.1f}s'>"
                f"<circle cx='{x:.0f}' cy='{y:.0f}' r='11' fill='#5B6BFF' fill-opacity='.16'/>"
                f"<circle cx='{x:.0f}' cy='{y:.0f}' r='3.4' fill='#9FE7FF'/></g>"
            )
        else:
            nodes.append(
                f"<circle cx='{x:.0f}' cy='{y:.0f}' r='{rnd.uniform(1.4, 2.4):.1f}' "
                f"fill='#BFD0FF' fill-opacity='.7'/>"
            )

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='xMidYMid slice'>"
        "<style>.t{animation:tw 5s ease-in-out infinite}"
        "@keyframes tw{0%,100%{opacity:.35}50%{opacity:1}}"
        "@media (prefers-reduced-motion:reduce){.t{animation:none}}</style>"
        "<defs><pattern id='grid' width='64' height='64' patternUnits='userSpaceOnUse'>"
        "<path d='M64 0H0V64' fill='none' stroke='#ffffff' stroke-opacity='.045'/></pattern></defs>"
        "<rect width='100%' height='100%' fill='url(#grid)'/>"
        + "".join(lines)
        + "".join(nodes)
        + "</svg>"
    )


def background_css() -> str:
    svg_uri = "data:image/svg+xml;utf8," + quote(build_network_svg())
    layers = ", ".join(
        [
            f'url("{svg_uri}") center / cover no-repeat fixed',
            "radial-gradient(60% 55% at 12% 8%, rgba(91,107,255,.50), transparent 62%) fixed",
            "radial-gradient(50% 45% at 88% 18%, rgba(139,92,246,.38), transparent 62%) fixed",
            "radial-gradient(55% 50% at 72% 95%, rgba(34,211,238,.24), transparent 62%) fixed",
            "linear-gradient(180deg, #0B1A3A 0%, #070E20 100%) fixed",
        ]
    )
    return (
        "<style>"
        f".stApp{{background:{layers} !important;}}"
        '[data-testid="stAppViewContainer"],[data-testid="stMain"]'
        "{background:transparent !important;}"
        "</style>"
    )


st.markdown(background_css(), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SEARCH_SNIPPET_LIMIT = 800  # same truncation used in pipeline.py

# (name, what it does, message shown while it runs)
STEPS = [
    ("Search", "Finds recent, reliable sources", "The search agent is looking for sources..."),
    ("Read", "Opens and scrapes the best one", "The reader agent is scraping the most relevant page..."),
    ("Write", "Drafts the report", "The writer is turning the research into a report..."),
    ("Review", "Critiques the draft", "The critic is reviewing the report..."),
]

EXAMPLE_TOPICS = [
    "Solid-state batteries",
    "GLP-1 drugs and the food industry",
    "Open-source language models",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def to_text(value) -> str:
    """Convert whatever a LangChain agent/chain returns into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):  # AIMessage etc.
        return to_text(value.content)
    if isinstance(value, list):  # list of content blocks
        parts = []
        for block in value:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(to_text(block))
        return "\n".join(p for p in parts if p)
    return str(value)


def tracker_html(current: int, timings: dict, failed: bool = False) -> str:
    """
    current: -1 = nothing started, 0-3 = that step is running, 4 = all finished.
    Steps before current are done.
    """
    stations = []
    for i, (name, desc, _) in enumerate(STEPS):
        if i < current:
            cls, mark = "done", "&#10003;"
        elif i == current:
            cls, mark = ("error", "!") if failed else ("active", str(i + 1))
        else:
            cls, mark = "idle", str(i + 1)
        took = f"<em>{timings[name]:.1f}s</em>" if (cls == "done" and name in timings) else ""
        stations.append(
            f'<div class="stn {cls}"><div class="dot">{mark}</div>'
            f"<b>{name}</b><span>{desc}</span>{took}</div>"
        )
    return '<div class="tracker">' + "".join(stations) + "</div>"


def run_pipeline(topic: str, tracker_slot, note_slot) -> dict:
    """Run the four steps and update the tracker as each one starts and ends."""
    state: dict = {"topic": topic}
    timings: dict = {}
    current = 0

    def begin(i: int) -> None:
        nonlocal current
        current = i
        tracker_slot.markdown(tracker_html(i, timings), unsafe_allow_html=True)
        note_slot.markdown(f'<div class="run-note">{STEPS[i][2]}</div>', unsafe_allow_html=True)

    try:
        # Step 1 - search
        begin(0)
        t0 = time.time()
        search_result = build_search_agent().invoke(
            {
                "messages": [
                    ("user", f"Find recent, reliable, and detailed information about: {topic}")
                ]
            }
        )
        state["search_result"] = to_text(search_result["messages"][-1].content)
        timings["Search"] = time.time() - t0

        # Step 2 - read
        begin(1)
        t0 = time.time()
        reader_result = build_reader_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_result'][:SEARCH_SNIPPET_LIMIT]}",
                    )
                ]
            }
        )
        state["scraped_content"] = to_text(reader_result["messages"][-1].content)
        timings["Read"] = time.time() - t0

        # Step 3 - write
        begin(2)
        t0 = time.time()
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_result']}\n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = to_text(
            writer_chain.invoke({"topic": topic, "research": research_combined})
        )
        timings["Write"] = time.time() - t0

        # Step 4 - review
        begin(3)
        t0 = time.time()
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        timings["Review"] = time.time() - t0

    except Exception:
        tracker_slot.markdown(tracker_html(current, timings, failed=True), unsafe_allow_html=True)
        note_slot.empty()
        raise

    tracker_slot.markdown(tracker_html(4, timings), unsafe_allow_html=True)
    note_slot.empty()
    state["timings"] = timings
    return state


def set_topic(value: str) -> None:
    st.session_state.topic_input = value


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("topic_input", "")
result = st.session_state.result

# --------------------------------------------------------------------------
# Hero: title, live tracker, input
# --------------------------------------------------------------------------
with st.container(key="hero"):
    st.markdown(
        '<h1 class="hero-title">Give a topic.<br>Get a reviewed report.</h1>'
        '<p class="hero-sub">One agent searches, one reads the best source, '
        "one writes the report and one critiques it.</p>",
        unsafe_allow_html=True,
    )

    tracker_slot = st.empty()
    note_slot = st.empty()
    if result:
        tracker_slot.markdown(tracker_html(4, result["timings"]), unsafe_allow_html=True)
    else:
        tracker_slot.markdown(tracker_html(-1, {}), unsafe_allow_html=True)

    with st.form("research_form", clear_on_submit=False):
        field_col, button_col = st.columns([5, 1.4], vertical_alignment="center")
        with field_col:
            topic = st.text_input(
                "Research topic",
                key="topic_input",
                placeholder="What do you want to research?",
                label_visibility="collapsed",
            )
        with button_col:
            submitted = st.form_submit_button("Run research")

    st.markdown('<div class="try-line">Or start from an example</div>', unsafe_allow_html=True)
    with st.container(key="chips"):
        chip_cols = st.columns(len(EXAMPLE_TOPICS))
        for col, example in zip(chip_cols, EXAMPLE_TOPICS):
            with col:
                st.button(example, key=f"chip_{example}", on_click=set_topic, args=(example,))

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if submitted:
    if not topic.strip():
        st.warning("Enter a research topic, then select Run research.")
    else:
        st.session_state.result = None
        try:
            st.session_state.result = run_pipeline(topic.strip(), tracker_slot, note_slot)
            result = st.session_state.result
        except Exception as exc:
            result = None
            st.error(f"The run stopped at the step marked in red: {exc}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if result:
    total = sum(result["timings"].values())
    st.markdown(
        f'<div class="result-meta">Report on <b>{html.escape(result["topic"])}</b>, '
        f"finished in {total:.0f} seconds.</div>",
        unsafe_allow_html=True,
    )

    report_col, notes_col = st.columns([1.8, 1], gap="large")

    with report_col:
        with st.container(key="report"):
            st.markdown(result["report"])
            st.download_button(
                "Download report (.md)",
                data=result["report"],
                file_name="research_report.md",
                mime="text/markdown",
            )

    with notes_col:
        with st.container(key="notes"):
            st.markdown('<h3 class="notes-title">Critic\'s review</h3>', unsafe_allow_html=True)
            st.markdown(result["feedback"])

    with st.container(key="sources"):
        with st.expander("Search results"):
            st.markdown(result["search_result"])
        with st.expander("Scraped page content"):
            st.markdown(result["scraped_content"])