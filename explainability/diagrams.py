# explainability/diagrams.py

from pyvis.network import Network
import streamlit.components.v1 as components

COLORS = {
    "input": "#AED6F1",
    "process": "#F9E79F",
    "decision": "#F5B7B1",
    "output": "#ABEBC6",
}

def render_pyvis(net, height=600):
    """
    Safe Streamlit renderer for PyVis.
    Avoids .show() and .write_html() entirely.
    """
    html = net.generate_html()
    components.html(html, height=height, scrolling=True)

def build_clean_bag_of_phrases_network(x_offset=0):
    net = Network(directed=True)

    net.add_node(
        "clean.input.text",
        label="text",
        color=COLORS["input"],
        shape="box",
        x=x_offset,
        y=0,
        fixed=True,
    )
    net.add_node(
        "clean.check.type",
        label="is list / str?",
        color=COLORS["decision"],
        shape="diamond",
        x=x_offset + 200,
        y=0,
        fixed=True,
    )
    net.add_node(
        "clean.join",
        label="join tokens",
        color=COLORS["process"],
        shape="box",
        x=x_offset + 400,
        y=-80,
        fixed=True,
    )
    net.add_node(
        "clean.normalize",
        label="normalize string",
        color=COLORS["process"],
        shape="box",
        x=x_offset + 400,
        y=80,
        fixed=True,
    )
    net.add_node(
        "clean.output",
        label="cleaned_text",
        color=COLORS["output"],
        shape="box",
        x=x_offset + 600,
        y=0,
        fixed=True,
    )

    net.add_edge("clean.input.text", "clean.check.type")
    net.add_edge("clean.check.type", "clean.join")
    net.add_edge("clean.check.type", "clean.normalize")
    net.add_edge("clean.join", "clean.output")
    net.add_edge("clean.normalize", "clean.output")

    return net

def build_analyze_popularity_network(x_offset=0):
    net = Network(directed=True)

    net.add_node(
        "pop.input.df",
        label="df",
        color=COLORS["input"],
        shape="box",
        x=x_offset,
        y=0,
        fixed=True,
    )
    net.add_node(
        "pop.input.metric",
        label="metric",
        color=COLORS["input"],
        shape="box",
        x=x_offset,
        y=120,
        fixed=True,
    )
    net.add_node(
        "pop.extract.tokens",
        label="extract tokens/entities",
        color=COLORS["process"],
        shape="box",
        x=x_offset + 250,
        y=0,
        fixed=True,
    )
    net.add_node(
        "pop.count.freq",
        label="count frequencies",
        color=COLORS["process"],
        shape="box",
        x=x_offset + 500,
        y=0,
        fixed=True,
    )
    net.add_node(
        "pop.rank.top",
        label="top/bottom tweets",
        color=COLORS["process"],
        shape="box",
        x=x_offset + 500,
        y=120,
        fixed=True,
    )
    net.add_node(
        "pop.output",
        label="tokens + tweet sets",
        color=COLORS["output"],
        shape="box",
        x=x_offset + 750,
        y=60,
        fixed=True,
    )

    net.add_edge("pop.input.df", "pop.extract.tokens")
    net.add_edge("pop.extract.tokens", "pop.count.freq")
    net.add_edge("pop.count.freq", "pop.output")
    net.add_edge("pop.input.metric", "pop.rank.top")
    net.add_edge("pop.rank.top", "pop.output")

    return net

def build_find_k_plexes_network(x_offset=0):
    net = Network(directed=True)

    net.add_node(
        "kplex.input.graph",
        label="graph",
        color=COLORS["input"],
        shape="box",
        x=x_offset,
        y=0,
        fixed=True,
    )
    net.add_node(
        "kplex.input.k",
        label="k",
        color=COLORS["input"],
        shape="box",
        x=x_offset,
        y=120,
        fixed=True,
    )
    net.add_node(
        "kplex.search",
        label="Bron–Kerbosch search",
        color=COLORS["process"],
        shape="box",
        x=x_offset + 250,
        y=60,
        fixed=True,
    )
    net.add_node(
        "kplex.check",
        label="is_k_plex",
        color=COLORS["decision"],
        shape="diamond",
        x=x_offset + 500,
        y=60,
        fixed=True,
    )
    net.add_node(
        "kplex.output",
        label="k-plexes",
        color=COLORS["output"],
        shape="box",
        x=x_offset + 750,
        y=60,
        fixed=True,
    )

    net.add_edge("kplex.input.graph", "kplex.search")
    net.add_edge("kplex.input.k", "kplex.search")
    net.add_edge("kplex.search", "kplex.check")
    net.add_edge("kplex.check", "kplex.output")

    return net
