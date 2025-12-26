from pyvis.network import Network

COLORS = {
    "input": "#AED6F1",
    "process": "#F9E79F",
    "decision": "#F5B7B1",
    "output": "#ABEBC6",
}

# --------------------------------------------------
# Streamlit-safe renderer
# --------------------------------------------------
def render_pyvis(net, height=700):
    import streamlit.components.v1 as components
    components.html(net.generate_html(), height=height, scrolling=True)


# --------------------------------------------------
# Step 1 — Load CSV
# --------------------------------------------------
def build_step_1_network(x_offset=0):
    net = Network(directed=True)
    net.toggle_physics(False)

    net.add_node("pN.step1.path", "csv_path", color=COLORS["input"],
                 shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("pN.step1.df", "raw_df", color=COLORS["output"],
                 shape="box", x=x_offset + 250, y=0, fixed=True)

    net.add_edge("pN.step1.path", "pN.step1.df")
    return net


# --------------------------------------------------
# Step 2 — JSON Parsing
# --------------------------------------------------
def build_step_2_network(x_offset=0):
    net = Network(directed=True)
    net.toggle_physics(False)

    net.add_node("pN.step2.text", "text", color=COLORS["input"],
                 shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("pN.step2.json", "parsed_json", color=COLORS["output"],
                 shape="box", x=x_offset + 250, y=0, fixed=True)

    net.add_edge("pN.step2.text", "pN.step2.json")
    return net


# --------------------------------------------------
# Step 3 — Explosion
# --------------------------------------------------
def build_step_3_network(x_offset=0):
    net = Network(directed=True)
    net.toggle_physics(False)

    net.add_node("pN.step3.df", "raw_df", color=COLORS["input"],
                 shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("pN.step3.exploded", "sentence_df", color=COLORS["output"],
                 shape="box", x=x_offset + 250, y=0, fixed=True)

    net.add_edge("pN.step3.df", "pN.step3.exploded")
    return net


# --------------------------------------------------
# Step 4 — Analysis & Audit
# --------------------------------------------------
def build_step_4_network(x_offset=0):
    net = Network(directed=True)
    net.toggle_physics(False)

    net.add_node("pN.step4.sentences", "sentence_df", color=COLORS["input"],
                 shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("pN.step4.audit", "coverage / grounding", color=COLORS["output"],
                 shape="box", x=x_offset + 300, y=0, fixed=True)

    net.add_edge("pN.step4.sentences", "pN.step4.audit")
    return net


# --------------------------------------------------
# TRUE Combined DAG
# --------------------------------------------------
def build_combined_network():
    net = Network(directed=True)
    net.toggle_physics(False)

    s1 = build_step_1_network(0)
    s2 = build_step_2_network(400)
    s3 = build_step_3_network(800)
    s4 = build_step_4_network(1200)

    for sub in [s1, s2, s3, s4]:
        for n in sub.nodes:
            net.add_node(**n)
        for e in sub.edges:
            net.add_edge(e["from"], e["to"])

    net.add_edge("pN.step1.df", "pN.step3.df", label="raw data")
    net.add_edge("pN.step2.json", "pN.step3.exploded", label="parsed ESG")
    net.add_edge("pN.step3.exploded", "pN.step4.sentences", label="sentences")

    return net
