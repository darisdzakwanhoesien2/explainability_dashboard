from pyvis.network import Network

COLORS = {
    "input": "#AED6F1",
    "process": "#F9E79F",
    "decision": "#F5B7B1",
    "output": "#ABEBC6",
}

# =====================================================
# Shared renderer
# =====================================================

def render_pyvis(net, height=700):
    import streamlit.components.v1 as components
    html = net.generate_html()
    components.html(html, height=height, scrolling=True)


# =====================================================
# Step 1 — Text preprocessing
# =====================================================

def build_step_1_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step1.input.raw_speech", label="raw speech text", color=COLORS["input"], shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("p32.step1.process.tokenize", label="tokenize + lowercase", color=COLORS["process"], shape="box", x=x_offset+250, y=0, fixed=True)
    net.add_node("p32.step1.process.clean", label="remove stopwords & punctuation", color=COLORS["process"], shape="box", x=x_offset+500, y=0, fixed=True)
    net.add_node("p32.step1.process.normalize", label="stem + lemmatize", color=COLORS["process"], shape="box", x=x_offset+750, y=0, fixed=True)
    net.add_node("p32.step1.output.tokens", label="processed tokens", color=COLORS["output"], shape="box", x=x_offset+1000, y=0, fixed=True)

    net.add_edge("p32.step1.input.raw_speech", "p32.step1.process.tokenize")
    net.add_edge("p32.step1.process.tokenize", "p32.step1.process.clean")
    net.add_edge("p32.step1.process.clean", "p32.step1.process.normalize")
    net.add_edge("p32.step1.process.normalize", "p32.step1.output.tokens")
    return net


# =====================================================
# Step 2 — Exploratory data analysis
# =====================================================

def build_step_2_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step2.input.metadata", label="speech metadata", color=COLORS["input"], shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("p32.step2.process.distributions", label="categorical & time distributions", color=COLORS["process"], shape="box", x=x_offset+300, y=0, fixed=True)
    net.add_node("p32.step2.output.plots", label="EDA plots", color=COLORS["output"], shape="box", x=x_offset+600, y=0, fixed=True)

    net.add_edge("p32.step2.input.metadata", "p32.step2.process.distributions")
    net.add_edge("p32.step2.process.distributions", "p32.step2.output.plots")
    return net


# =====================================================
# Step 3 — Sentiment normalization & categorization
# =====================================================

def build_step_3_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step3.input.sentiment_scores", label="raw sentiment metrics", color=COLORS["input"], shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("p32.step3.process.znorm", label="z-normalization", color=COLORS["process"], shape="box", x=x_offset+300, y=0, fixed=True)
    net.add_node("p32.step3.process.weighting", label="weighted averaging", color=COLORS["process"], shape="box", x=x_offset+600, y=0, fixed=True)
    net.add_node("p32.step3.decision.binning", label="sentiment binning", color=COLORS["decision"], shape="diamond", x=x_offset+900, y=0, fixed=True)
    net.add_node("p32.step3.output.category", label="sentiment category", color=COLORS["output"], shape="box", x=x_offset+1200, y=0, fixed=True)

    net.add_edge("p32.step3.input.sentiment_scores", "p32.step3.process.znorm")
    net.add_edge("p32.step3.process.znorm", "p32.step3.process.weighting")
    net.add_edge("p32.step3.process.weighting", "p32.step3.decision.binning")
    net.add_edge("p32.step3.decision.binning", "p32.step3.output.category")
    return net


# =====================================================
# Step 4 — Word frequency & n-grams
# =====================================================

def build_step_4_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step4.input.tokens", label="processed tokens", color=COLORS["input"], shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("p32.step4.process.freq", label="frequency & n-grams", color=COLORS["process"], shape="box", x=x_offset+300, y=0, fixed=True)
    net.add_node("p32.step4.output.wordclouds", label="word clouds / bar charts", color=COLORS["output"], shape="box", x=x_offset+600, y=0, fixed=True)

    net.add_edge("p32.step4.input.tokens", "p32.step4.process.freq")
    net.add_edge("p32.step4.process.freq", "p32.step4.output.wordclouds")
    return net


# =====================================================
# Step 5 — Similarity computation
# =====================================================

def build_step_5_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step5.input.tokens", label="processed tokens", color=COLORS["input"], shape="box", x=x_offset, y=-100, fixed=True)
    net.add_node("p32.step5.input.demographics", label="demographic features", color=COLORS["input"], shape="box", x=x_offset, y=100, fixed=True)
    net.add_node("p32.step5.process.tfidf", label="TF-IDF similarity", color=COLORS["process"], shape="box", x=x_offset+350, y=-100, fixed=True)
    net.add_node("p32.step5.process.doc2vec", label="Doc2Vec similarity", color=COLORS["process"], shape="box", x=x_offset+350, y=0, fixed=True)
    net.add_node("p32.step5.process.demo_sim", label="demographic similarity", color=COLORS["process"], shape="box", x=x_offset+350, y=100, fixed=True)
    net.add_node("p32.step5.output.similarity", label="similarity matrices", color=COLORS["output"], shape="box", x=x_offset+700, y=0, fixed=True)

    net.add_edge("p32.step5.input.tokens", "p32.step5.process.tfidf")
    net.add_edge("p32.step5.input.tokens", "p32.step5.process.doc2vec")
    net.add_edge("p32.step5.input.demographics", "p32.step5.process.demo_sim")
    net.add_edge("p32.step5.process.tfidf", "p32.step5.output.similarity")
    net.add_edge("p32.step5.process.doc2vec", "p32.step5.output.similarity")
    net.add_edge("p32.step5.process.demo_sim", "p32.step5.output.similarity")
    return net


# =====================================================
# Step 6 — Clustering
# =====================================================

def build_step_6_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step6.input.similarity", label="similarity matrices", color=COLORS["input"], shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("p32.step6.process.kmeans", label="KMeans", color=COLORS["process"], shape="box", x=x_offset+300, y=-100, fixed=True)
    net.add_node("p32.step6.process.dbscan", label="DBSCAN", color=COLORS["process"], shape="box", x=x_offset+300, y=0, fixed=True)
    net.add_node("p32.step6.process.agglomerative", label="Agglomerative", color=COLORS["process"], shape="box", x=x_offset+300, y=100, fixed=True)
    net.add_node("p32.step6.output.clusters", label="cluster labels", color=COLORS["output"], shape="box", x=x_offset+650, y=0, fixed=True)

    net.add_edge("p32.step6.input.similarity", "p32.step6.process.kmeans")
    net.add_edge("p32.step6.input.similarity", "p32.step6.process.dbscan")
    net.add_edge("p32.step6.input.similarity", "p32.step6.process.agglomerative")
    net.add_edge("p32.step6.process.kmeans", "p32.step6.output.clusters")
    net.add_edge("p32.step6.process.dbscan", "p32.step6.output.clusters")
    net.add_edge("p32.step6.process.agglomerative", "p32.step6.output.clusters")
    return net


# =====================================================
# Step 7 — Topic modeling & evolution
# =====================================================

def build_step_7_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step7.input.clusters", label="clustered speeches", color=COLORS["input"], shape="box", x=x_offset, y=0, fixed=True)
    net.add_node("p32.step7.process.lda", label="LDA topics", color=COLORS["process"], shape="box", x=x_offset+300, y=-50, fixed=True)
    net.add_node("p32.step7.process.bertopic", label="BERTopic (time)", color=COLORS["process"], shape="box", x=x_offset+300, y=50, fixed=True)
    net.add_node("p32.step7.output.trends", label="topic trends", color=COLORS["output"], shape="box", x=x_offset+650, y=0, fixed=True)

    net.add_edge("p32.step7.input.clusters", "p32.step7.process.lda")
    net.add_edge("p32.step7.input.clusters", "p32.step7.process.bertopic")
    net.add_edge("p32.step7.process.lda", "p32.step7.output.trends")
    net.add_edge("p32.step7.process.bertopic", "p32.step7.output.trends")
    return net


# =====================================================
# Step 8 — NER & emotion prediction
# =====================================================

def build_step_8_network(x_offset=0):
    net = Network(height="600px", width="100%", directed=True)
    net.add_node("p32.step8.input.speech", label="speech text", color=COLORS["input"], shape="box", x=x_offset, y=-50, fixed=True)
    net.add_node("p32.step8.process.ner", label="spaCy NER", color=COLORS["process"], shape="box", x=x_offset+300, y=-50, fixed=True)
    net.add_node("p32.step8.process.ml", label="ML / Transformer models", color=COLORS["process"], shape="box", x=x_offset+300, y=50, fixed=True)
    net.add_node("p32.step8.output.entities", label="entity networks", color=COLORS["output"], shape="box", x=x_offset+650, y=-50, fixed=True)
    net.add_node("p32.step8.output.predictions", label="emotion predictions", color=COLORS["output"], shape="box", x=x_offset+650, y=50, fixed=True)

    net.add_edge("p32.step8.input.speech", "p32.step8.process.ner")
    net.add_edge("p32.step8.input.speech", "p32.step8.process.ml")
    net.add_edge("p32.step8.process.ner", "p32.step8.output.entities")
    net.add_edge("p32.step8.process.ml", "p32.step8.output.predictions")
    return net


# =====================================================
# Combined end-to-end DAG
# =====================================================

def build_combined_network():
    net = Network(height="800px", width="100%", directed=True)

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
                fixed=True,
                color=n.get("color"),
                shape=n.get("shape"),
            )
        for e in source_net.edges:
            net.add_edge(e["from"], e["to"], label=e.get("label"))

    steps = [
        build_step_1_network(0),
        build_step_2_network(1400),
        build_step_3_network(2800),
        build_step_4_network(4200),
        build_step_5_network(5600),
        build_step_6_network(7200),
        build_step_7_network(8800),
        build_step_8_network(10400),
    ]

    for s in steps:
        add_nodes_from(s)

    net.add_edge("p32.step1.output.tokens", "p32.step4.input.tokens", label="semantic flow")
    net.add_edge("p32.step3.output.category", "p32.step4.process.freq", label="sentiment conditioning")
    net.add_edge("p32.step5.output.similarity", "p32.step6.input.similarity", label="semantic flow")
    net.add_edge("p32.step6.output.clusters", "p32.step7.input.clusters", label="semantic flow")
    net.add_edge("p32.step7.output.trends", "p32.step8.input.speech", label="context flow")

    return net
