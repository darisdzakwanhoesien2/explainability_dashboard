# explainability/diagrams.py
from pyvis.network import Network

# --------------------------------------------------
# Shared renderer
# --------------------------------------------------
def render_pyvis(net):
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()
    os.unlink(tmp.name)
    return html


# --------------------------------------------------
# Weakness extraction diagram
# --------------------------------------------------
def build_weakness_network(x_offset=0):
    net = Network(height="650px", width="100%", directed=True)
    net.toggle_physics(False)

    COLORS = {
        "input": "#97C2FC",
        "process": "#D3D3D3",
        "decision": "#FFB347",
        "output": "#77DD77",
    }

    nodes = [
        ("w_ref_xy", "ref_x, ref_y", "input", 0, 180),
        ("w_drv_xy", "drv_x, drv_y", "input", 0, 320),

        ("w_distance", "compute distance\n(dist, idx)", "process", 300, 260),
        ("w_kappa", "curvature\n(kappa)", "process", 300, 120),

        ("w_apex_mask", "apex mask\n(kappa > p75)", "decision", 600, 120),

        ("w_apex", "w_apex", "process", 900, 60),
        ("w_slalom", "w_slalom", "process", 900, 220),
        ("w_corner", "w_corner", "process", 900, 380),

        ("weakness_vec", "weakness_vec", "output", 1200, 220),
    ]

    for node_id, label, typ, x, y in nodes:
        net.add_node(
            node_id,
            label=label,
            title=label,
            x=x + x_offset,
            y=y,
            fixed=True,
            color=COLORS[typ],
            shape="box" if typ != "decision" else "diamond",
        )

    edges = [
        ("w_ref_xy", "w_distance"),
        ("w_drv_xy", "w_distance"),
        ("w_ref_xy", "w_kappa"),
        ("w_kappa", "w_apex_mask"),

        ("w_distance", "w_apex"),
        ("w_apex_mask", "w_apex"),

        ("w_distance", "w_slalom"),
        ("w_distance", "w_corner"),

        ("w_apex", "weakness_vec"),
        ("w_slalom", "weakness_vec"),
        ("w_corner", "weakness_vec"),
    ]

    for src, dst in edges:
        net.add_edge(src, dst, arrows="to")

    return net



def build_deformation_network(x_offset=0):
    net = Network(height="720px", width="100%", directed=True)
    net.toggle_physics(False)

    COLORS = {
        "input": "#97C2FC",
        "process": "#D3D3D3",
        "decision": "#FFB347",
        "output": "#77DD77",
    }

    nodes = [
        ("d_ref_xy", "ref_x, ref_y", "input", 0, 180),
        ("weakness_vec", "weakness_vec", "input", 0, 420),
        ("d_max_offset", "max_offset", "input", 0, 260),

        ("d_arc_length", "arc length (s)", "process", 300, 120),
        ("d_s_norm", "s_norm", "process", 600, 120),
        ("d_normals", "compute normals", "process", 300, 260),
        ("d_kappa", "curvature", "process", 300, 420),

        ("d_argmax", "argmax", "decision", 300, 560),
        ("d_focus", "focus", "decision", 600, 560),
        ("d_strength", "strength", "process", 600, 420),

        ("d_mask", "mask", "decision", 900, 440),

        ("d_delta_raw", "delta_raw", "process", 900, 180),
        ("d_delta_smooth", "delta_smooth", "process", 1200, 180),

        ("new_x", "new_x", "output", 1500, 120),
        ("new_y", "new_y", "output", 1500, 220),
    ]

    for node_id, label, typ, x, y in nodes:
        net.add_node(
            node_id,
            label=label,
            title=label,
            x=x + x_offset,
            y=y,
            fixed=True,
            color=COLORS[typ],
            shape="box" if typ != "decision" else "diamond",
        )

    edges = [
        ("d_ref_xy", "d_arc_length"),
        ("d_arc_length", "d_s_norm"),
        ("d_ref_xy", "d_normals"),
        ("d_ref_xy", "d_kappa"),

        ("weakness_vec", "d_argmax"),
        ("d_argmax", "d_focus"),
        ("weakness_vec", "d_strength"),

        ("d_s_norm", "d_delta_raw"),
        ("d_strength", "d_delta_raw"),
        ("d_max_offset", "d_delta_raw"),
        ("d_mask", "d_delta_raw"),

        ("d_delta_raw", "d_delta_smooth"),

        ("d_delta_smooth", "new_x"),
        ("d_normals", "new_x"),
        ("d_delta_smooth", "new_y"),
        ("d_normals", "new_y"),
    ]

    for src, dst in edges:
        net.add_edge(src, dst, arrows="to")

    return net


def build_combined_network():
    net = Network(height="800px", width="100%", directed=True)
    net.toggle_physics(False)

    # Build subgraphs
    w_net = build_weakness_network(x_offset=0)
    d_net = build_deformation_network(x_offset=1500)

    # --------------------------------------------------
    # Add nodes properly (CRITICAL FIX)
    # --------------------------------------------------
    def add_nodes_from(source_net, skip_ids=None):
        skip_ids = skip_ids or set()
        for n in source_net.nodes:
            if n["id"] in skip_ids:
                continue
            net.add_node(
                n["id"],
                label=n.get("label"),
                title=n.get("title"),
                x=n.get("x"),
                y=n.get("y"),
                fixed=n.get("fixed", True),
                color=n.get("color"),
                shape=n.get("shape"),
            )

    add_nodes_from(w_net)
    add_nodes_from(d_net, skip_ids={"weakness_vec"})

    # --------------------------------------------------
    # Add edges properly
    # --------------------------------------------------
    for e in w_net.edges:
        net.add_edge(e["from"], e["to"], arrows="to")

    for e in d_net.edges:
        net.add_edge(e["from"], e["to"], arrows="to")

    # --------------------------------------------------
    # Explicit semantic links
    # --------------------------------------------------
    net.add_edge("weakness_vec", "d_argmax", arrows="to")
    net.add_edge("weakness_vec", "d_strength", arrows="to")

    return net
