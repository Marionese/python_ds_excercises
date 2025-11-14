import ast
from collections import defaultdict, Counter

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from typing import Optional


# ---------- Helpers ----------

def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        st.error("Please upload a CSV or JSON file.")
        return None

    # Convert stringified lists back into real Python lists
    list_cols = ["castList", "directorList", "genreList", "countryList", "characterList"]
    for col in list_cols:
        if col in df.columns and df[col].dtype == object:
            def parse_maybe_list(x):
                if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                    try:
                        return ast.literal_eval(x)
                    except Exception:
                        return x
                return x

            df[col] = df[col].apply(parse_maybe_list)

    return df


def minutes_to_h_m(minutes) -> str:
    if pd.isna(minutes):
        return "0 min"
    minutes = int(minutes)
    h = minutes // 60
    m = minutes % 60
    if h == 0:
        return f"{m} min"
    return f"{h} h {m} min"


def aggregate_actor_metrics(df: pd.DataFrame):
    duration_by_actor = defaultdict(int)
    gross_by_actor = defaultdict(int)
    count_by_actor = defaultdict(int)

    for _, row in df.iterrows():
        cast = row.get("castList", [])
        if not isinstance(cast, list):
            continue

        dur = row.get("duration", 0) or 0
        gross = row.get("gross", 0) or 0

        dur = int(dur)
        gross = int(gross)

        for actor in cast:
            duration_by_actor[actor] += dur
            gross_by_actor[actor] += gross
            count_by_actor[actor] += 1

    return duration_by_actor, gross_by_actor, count_by_actor


def build_collaboration_edges(df: pd.DataFrame):
    edge_weights = Counter()

    for _, row in df.iterrows():
        directors = row.get("directorList", [])
        cast = row.get("castList", [])

        if not isinstance(directors, list) or not isinstance(cast, list):
            continue

        for d in directors:
            for a in cast:
                edge_weights[(d, a)] += 1

    return edge_weights


def make_pyvis_graph(edge_weights: Counter, top_n: int) -> str:
    sorted_edges = sorted(
        edge_weights.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
    )[:top_n]

    net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")
    net.barnes_hut()

    added_nodes = set()

    for (director, actor), weight in sorted_edges:
        if director not in added_nodes:
            net.add_node(director, label=director, title=f"Director: {director}", group="director")
            added_nodes.add(director)

        if actor not in added_nodes:
            net.add_node(actor, label=actor, title=f"Actor: {actor}", group="actor")
            added_nodes.add(actor)

        net.add_edge(director, actor, value=weight, title=f"{weight} movie(s) together")

    # 👉 valid JSON string (no 'const options =', keys quoted)
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 8,
        "borderWidth": 1,
        "font": { "size": 14 }
      },
      "edges": {
        "smooth": false,
        "width": 1,
        "color": { "opacity": 0.5 }
      },
      "physics": {
        "stabilization": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 150,
          "springConstant": 0.02
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    return net.generate_html("graph.html")



# ---------- Streamlit UI ----------

st.set_page_config(page_title="IMDb Top 250 Explorer", layout="wide")
st.title("🎬 Minimal IMDb Top 250 Explorer")
st.caption("Upload your Top250 CSV/JSON and explore patience, workhorses, cashhorses & collaborations.")

uploaded_file = st.file_uploader("Upload IMDb Top250 as CSV or JSON", type=["csv", "json"])
df = load_data(uploaded_file)

if df is None:
    st.info("Upload the file to get started.")
    st.stop()

st.subheader("Raw Data")
st.dataframe(df)

required_cols = ["title", "duration", "gross", "year", "castList", "directorList"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing expected columns: {', '.join(missing)}")
    st.stop()

duration_by_actor, gross_by_actor, count_by_actor = aggregate_actor_metrics(df)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Patience", "Binge Watching Steven Spielberg", "This is about me", "Workhorse", "Cashhorse"]
)

# ---------- Tabs ----------

with tab1:
    st.subheader("Patience ⏳")
    long_bois = df[df["duration"] >= 220].sort_values("duration", ascending=False)
    st.write(f"Found **{len(long_bois)}** movies with duration ≥ 220 minutes.")
    st.dataframe(long_bois[["title", "year", "duration", "ratingValue", "ratingCount"]])

with tab2:
    st.subheader("Binge Watching Steven Spielberg 🎥")

    def is_spielberg(row):
        dirs = row.get("directorList", [])
        return isinstance(dirs, list) and "Steven Spielberg" in dirs

    spielberg_df = df[df.apply(is_spielberg, axis=1)]
    total_minutes = int(spielberg_df["duration"].fillna(0).sum())

    st.metric("Total Spielberg running time", minutes_to_h_m(total_minutes))
    st.dataframe(spielberg_df[["title", "year", "duration", "ratingValue", "ratingCount"]])

with tab3:
    st.subheader("This is about me 🧍 (Screen-time)")

    actor_duration_df = pd.DataFrame(
        [{"actor": a, "total_minutes": m} for a, m in duration_by_actor.items()]
    ).sort_values("total_minutes", ascending=False)

    top10 = actor_duration_df.head(10)
    top10["total_h_m"] = top10["total_minutes"].apply(minutes_to_h_m)

    st.dataframe(top10.set_index("actor"))

with tab4:
    st.subheader("Workhorse 🐴")

    actor_count_df = pd.DataFrame(
        [{"actor": a, "movie_count": c} for a, c in count_by_actor.items()]
    ).sort_values("movie_count", ascending=False)

    st.dataframe(actor_count_df.head(10).set_index("actor"))

with tab5:
    st.subheader("Cashhorse 💰")

    actor_gross_df = pd.DataFrame(
        [{"actor": a, "total_gross": g} for a, g in gross_by_actor.items()]
    ).sort_values("total_gross", ascending=False)

    actor_gross_df["formatted"] = actor_gross_df["total_gross"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(actor_gross_df.head(10).set_index("actor")[["total_gross", "formatted"]])

# ---------- Extra Features ----------

st.markdown("---")
col_timeline, col_graph = st.columns(2)

with col_timeline:
    st.subheader("Timeline: Number of Movies per Year 📅")
    movies_per_year = df.groupby("year")["title"].count().reset_index(name="movie_count")
    st.line_chart(movies_per_year.set_index("year"))

with col_graph:
    st.subheader("Director–Actor Collaboration Network 🕸️")

    edges = build_collaboration_edges(df)
    if edges:
        total_edges = len(edges)
        min_edges = min(10, total_edges)
        top_n = st.slider("Number of strongest collaborations", min_edges, total_edges, min_edges)

        html = make_pyvis_graph(edges, top_n)
        components.html(html, height=600, scrolling=True)

st.caption("Powered by Streamlit, NetworkX, PyVis — and your IMDb Top250 dataset.")
