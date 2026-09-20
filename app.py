"""
The Metastasis Detective — an educational Streamlit app.
Explore how a machine-learning model classifies melanoma tumors as
primary or metastatic based on RNA gene expression.

Educational tool only — not a diagnostic device.
"""
import streamlit as st
import joblib
import numpy as np

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="The Metastasis Detective",
    page_icon="🧬",
    layout="wide",
)


# ----------------------------------------------------------------------
# Hide Streamlit's default toolbar (Share / star / GitHub / menu) for a cleaner look
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)
 

# ----------------------------------------------------------------------
# Load models once and cache them
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    tree = joblib.load("model_tree.pkl")
    logreg = joblib.load("model_logreg.pkl")
    rf = joblib.load("model_rf.pkl")
    xgb = joblib.load("model_xgb.pkl")
    meta = joblib.load("model_meta.pkl")
    try:
        examples = joblib.load("model_examples.pkl")
    except Exception:
        examples = {}
    return tree, logreg, rf, xgb, meta, examples

tree, logreg, rf, xgb, meta, EXAMPLES = load_models()
TOP100 = meta["top_100_genes"]
TREE_GENES = ["S100A7", "C7", "KRT17", "KLK5", "DEFB103B"]  # order for display
ACC = meta["accuracies"]
GENE_IDX = {g: i for i, g in enumerate(TOP100)}

# ----------------------------------------------------------------------
# Core logic: build model input + trace the decision path
# ----------------------------------------------------------------------
def build_input(slider_values):
    """95 non-slider genes default to 0 (z-score mean); 5 sliders override."""
    x = np.zeros(len(TOP100))
    for g, v in slider_values.items():
        x[GENE_IDX[g]] = v
    return x.reshape(1, -1)

def trace_path(slider_values):
    """Walk the decision tree, returning each split and the final prediction."""
    t = tree.tree_
    x = build_input(slider_values)[0]
    node, path = 0, []
    while t.feature[node] != -2:            # -2 marks a leaf
        gene = TOP100[t.feature[node]]
        thr = float(t.threshold[node])
        val = float(x[GENE_IDX[gene]])
        went_left = val <= thr
        path.append({"gene": gene, "thr": thr, "val": val, "left": went_left})
        node = t.children_left[node] if went_left else t.children_right[node]
    vals = t.value[node][0]
    pred = "Metastatic" if vals[1] >= vals[0] else "Primary"
    return path, pred

def predict_proba_quiet(model, x):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict_proba(x)[0]

# ----------------------------------------------------------------------
# Decision-tree diagram with the active path highlighted
# ----------------------------------------------------------------------
@st.cache_data
def _tree_layout():
    """Compute node positions once (cached)."""
    t = tree.tree_
    positions = {}
    def layout(node, depth, xmin, xmax):
        positions[node] = ((xmin + xmax) / 2, -depth)
        if t.feature[node] != -2:
            mid = (xmin + xmax) / 2
            layout(t.children_left[node], depth + 1, xmin, mid)
            layout(t.children_right[node], depth + 1, mid, xmax)
    layout(0, 0, 0, 1)
    return positions

def _active_leaf(slider_values):
    """Which leaf the current inputs reach — the cache key for the diagram."""
    t = tree.tree_
    x = np.zeros(len(TOP100))
    for g, v in slider_values.items():
        x[GENE_IDX[g]] = v
    node = 0
    while t.feature[node] != -2:
        gene = TOP100[t.feature[node]]
        node = (t.children_left[node] if x[GENE_IDX[gene]] <= t.threshold[node]
                else t.children_right[node])
    return node

@st.cache_data(max_entries=16)
def draw_tree(leaf_id):
    """Render the tree with the path to `leaf_id` highlighted. Cached per leaf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tree.tree_
    positions = _tree_layout()

    # reconstruct the active path from leaf back to root
    parent = {}
    for n in range(t.node_count):
        if t.feature[n] != -2:
            parent[t.children_left[n]] = n
            parent[t.children_right[n]] = n
    active = {leaf_id}
    cur = leaf_id
    while cur in parent:
        cur = parent[cur]
        active.add(cur)

    def label(n):
        if t.feature[n] != -2:
            return f"{TOP100[t.feature[n]]}\n\u2264 {t.threshold[n]:.2f}?"
        vals = t.value[n][0]
        return "Metastatic" if vals[1] >= vals[0] else "Primary"

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_alpha(0)
    for n in range(t.node_count):
        if t.feature[n] != -2:
            for child in (t.children_left[n], t.children_right[n]):
                x0, y0 = positions[n]; x1, y1 = positions[child]
                on = n in active and child in active
                ax.plot([x0, x1], [y0, y1], color="#555" if on else "#ddd",
                        lw=2.5 if on else 1, zorder=1)
    for n in range(t.node_count):
        px, py = positions[n]
        on = n in active
        if t.feature[n] == -2:
            vals = t.value[n][0]
            cls = "Metastatic" if vals[1] >= vals[0] else "Primary"
            base = "#c0392b" if cls == "Metastatic" else "#2c6fbb"
        else:
            base = "#7f8c8d"
        ax.text(px, py, label(n), ha="center", va="center", fontsize=9,
                color="white" if on else "#aaa", zorder=3,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=base if on else "#f0f0f0",
                          edgecolor="#333" if on else "#ccc",
                          linewidth=1.5 if on else 0.5))
    ax.axis("off")
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("The Metastasis Detective")
page = st.sidebar.radio(
    "Explore",
    ["Home", "Interactive Diagnosis", "Gene Stories", "Model Comparison"],
)
st.sidebar.caption("Educational tool — not for real diagnosis.")

# ======================================================================
# PAGE: HOME
# ======================================================================
if page == "Home":
    st.title("The Metastasis Detective")
    st.subheader("How does an AI read cancer's genetic signals?")
    st.write(
        "Melanoma is dangerous once it spreads (metastasizes). This app lets you "
        "step into the role of a detective: adjust the expression of a few genes and "
        "watch a real machine-learning model decide whether a tumor looks **primary** "
        "(still in its original spot) or **metastatic** (spread to a new location)."
    )
    st.write(
        "Behind the scenes runs a model trained on **real melanoma data** from the "
        "TCGA-SKCM project — the same public cancer dataset used in actual research."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Logistic Regression", f"{ACC['logreg']*100:.1f}%")
    c2.metric("Decision Tree", f"{ACC['tree']*100:.1f}%")
    c3.metric("Random Forest", f"{ACC['rf']*100:.1f}%")
    c4.metric("XGBoost", f"{ACC['xgb']*100:.1f}%")
    c5.metric("SVM", f"{ACC['svm']*100:.1f}%")
    st.info(
        "This is an **educational simulation** built to show how AI classification "
        "works. It is not a medical device and does not diagnose real patients.",
        icon="ℹ️",
    )
    st.write("<- Open **Interactive Diagnosis** in the sidebar to try it yourself.")

# ======================================================================
# PAGE: INTERACTIVE DIAGNOSIS  (the MVP core)
# ======================================================================
elif page == "Interactive Diagnosis":
    st.title("Interactive Diagnosis")
    st.write(
        "Adjust the five genes the decision tree actually uses. Values are "
        "**z-scores**: 0 is average expression, negative is below average, positive "
        "is above average."
    )

    # Presets
    st.write("**Quick start — load an example:**")
    # Real patient examples pulled from the TCGA dataset
    st.write("**Quick start — load a real patient from the TCGA dataset:**")
    ex_labels = list(EXAMPLES.keys())
    if "preset" not in st.session_state:
        st.session_state.preset = {"S100A7": 0.5, "C7": -1.5, "KRT17": -0.8,
                                    "KLK5": 0.0, "DEFB103B": 0.0}
        st.session_state.actual = None
    ex_cols = st.columns(len(ex_labels) + 1) if ex_labels else st.columns(1)
    for i, lbl in enumerate(ex_labels):
        if ex_cols[i].button(lbl):
            st.session_state.preset = dict(EXAMPLES[lbl]["genes"])
            st.session_state.actual = EXAMPLES[lbl]["actual"]
    if ex_cols[-1].button("Reset to average"):
        st.session_state.preset = {g: 0.0 for g in TREE_GENES}
        st.session_state.actual = None

    col_in, col_out = st.columns([3, 2])

    with col_in:
        st.subheader("Gene expression")
        vals = {}
        pre = st.session_state.preset
        for g in TREE_GENES:
            vals[g] = st.slider(g, -3.0, 3.0, float(pre.get(g, 0.0)), 0.1)

    path, pred = trace_path(vals)
    proba = predict_proba_quiet(tree, build_input(vals))
    conf = max(proba) * 100

    with col_out:
        st.subheader("Decision tree prediction")
        if pred == "Metastatic":
            st.error(f"### {pred}")
        else:
            st.success(f"### {pred}")
        st.progress(conf / 100)
        st.caption(f"{conf:.0f}% confidence")

        # If a real example is loaded, show the true label and whether it's correct
        actual = st.session_state.get("actual")
        if actual:
            if actual == pred:
                st.caption(f"✅ Actual diagnosis: **{actual}** — the tree got it right.")
            else:
                st.caption(f"❌ Actual diagnosis: **{actual}** — the tree got this one wrong.")

    st.divider()
    st.subheader("How the decision tree decided")
    st.caption("The highlighted path shows the route your gene values take through the tree.")
    st.pyplot(draw_tree(_active_leaf(vals)))

    st.markdown("**Step by step:**")
    for step in path:
        arrow = "≤" if step["left"] else ">"
        direction = "go left" if step["left"] else "go right"
        st.markdown(
            f"**{step['gene']}**: {step['val']:.2f} {arrow} {step['thr']:.2f} "
            f"→ *{direction}*"
        )
    st.markdown(f"### → {pred}")
    st.caption(
        "Each row is one question the tree asks. It checks a gene against a "
        "threshold, then moves down the branch until it reaches a conclusion. "
        "The tree uses only these 5 genes — that's why you only adjust 5 sliders."
    )

# ======================================================================
# PAGE: GENE STORIES
# ======================================================================
elif page == "Gene Stories":
    st.title("Gene Stories")
    st.write(
        "The decision tree relies on just five genes. Here's what each one is, the "
        "role it plays in the tree, and where to read more. These descriptions come "
        "from cancer research broadly — the app shows a pattern the model found in "
        "this dataset, not proof of what causes melanoma to spread."
    )

    GENE_INFO = {
        "S100A7": {
            "full": "S100A7 (psoriasin)",
            "what": "A calcium-binding protein made by skin keratinocytes, first "
                    "discovered in psoriasis. It's linked to inflammation, attracting "
                    "immune cells, and — in several epithelial skin tumors — is turned "
                    "up as tumors progress.",
            "role": "The tree's very first question. It splits the whole dataset in "
                    "two, making it the single most influential gene in the model.",
            "genecards": "https://www.genecards.org/cgi-bin/carddisp.pl?gene=S100A7",
            "wiki": "https://en.wikipedia.org/wiki/S100A7",
        },
        "C7": {
            "full": "C7 (complement component 7)",
            "what": "A protein of the complement system — part of the body's innate "
                    "immune defense. Its cancer link is complex: in some cancers lower "
                    "C7 tracks with worse outcomes, in others higher C7 tracks with "
                    "spread. That context-dependence makes it stand out.",
            "role": "The tree's second-tier question on both branches. It repeatedly "
                    "refines the decision after S100A7.",
            "genecards": "https://www.genecards.org/cgi-bin/carddisp.pl?gene=C7",
            "wiki": "https://en.wikipedia.org/wiki/Complement_component_7",
        },
        "KRT17": {
            "full": "KRT17 (keratin 17)",
            "what": "A structural filament protein in skin cells that also helps "
                    "regulate growth, immune response, and differentiation. In many "
                    "cancers higher KRT17 tracks with more proliferation and invasion, "
                    "though its role can flip by tumor type.",
            "role": "A lower-branch tiebreaker. It settles one specific primary-vs-"
                    "metastatic split near the bottom of the tree.",
            "genecards": "https://www.genecards.org/cgi-bin/carddisp.pl?gene=KRT17",
            "wiki": "https://en.wikipedia.org/wiki/Keratin_17",
        },
        "KLK5": {
            "full": "KLK5 (kallikrein-related peptidase 5)",
            "what": "An enzyme (serine protease) highly expressed in skin, where it "
                    "helps shed dead surface cells (desquamation). It's been reported "
                    "as differentially expressed in several cancers.",
            "role": "A lower-branch decision point on the left side of the tree, used "
                    "after S100A7 and C7 to reach a conclusion.",
            "genecards": "https://www.genecards.org/cgi-bin/carddisp.pl?gene=KLK5",
            "wiki": "https://en.wikipedia.org/wiki/KLK5",
        },
        "DEFB103B": {
            "full": "DEFB103B (beta-defensin 103B)",
            "what": "An antimicrobial peptide expressed in skin epithelium, part of "
                    "innate immune defense against microbes. It also helps bridge "
                    "innate and adaptive immunity.",
            "role": "The first question on the tree's right branch (when S100A7 is "
                    "high), steering toward a primary-vs-metastatic conclusion.",
            "genecards": "https://www.genecards.org/cgi-bin/carddisp.pl?gene=DEFB103B",
            "wiki": "https://en.wikipedia.org/wiki/DEFB103A",
        },
    }

    for gene in TREE_GENES:
        info = GENE_INFO[gene]
        with st.expander(info["full"]):
            st.markdown("**What it is**")
            st.write(info["what"])
            st.markdown("**Its role in the decision tree**")
            st.write(info["role"])
            st.markdown(
                f"**Read more:** [GeneCards]({info['genecards']}) · "
                f"[Wikipedia]({info['wiki']})"
            )

    st.caption(
        "Sources: GeneCards and Wikipedia for general gene information. Biological "
        "roles are drawn from published cancer research and reflect correlation the "
        "model found — not established causation in melanoma."
    )

# ======================================================================
# PAGE: MODEL COMPARISON
# ======================================================================
elif page == "Model Comparison":
    import pandas as pd

    st.title("Model Comparison")
    st.write(
        "The same melanoma data was analyzed three different ways. Each algorithm "
        "reaches its answer differently — and they don't fully agree on which genes "
        "matter most. Comparing them is part of understanding how AI works."
    )

    # --- Accuracy row ---
    st.subheader("Accuracy on held-out test data")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Logistic Regression", f"{ACC['logreg']*100:.1f}%")
    c2.metric("Decision Tree", f"{ACC['tree']*100:.1f}%")
    c3.metric("Random Forest", f"{ACC['rf']*100:.1f}%")
    c4.metric("XGBoost", f"{ACC['xgb']*100:.1f}%")
    c5.metric("SVM", f"{ACC['svm']*100:.1f}%")
    st.caption(
        "Higher isn't automatically 'better' — a simple model that's easy to explain "
        "can be more useful than a complex one that's slightly more accurate. "
        "SVM is scored with 5-fold cross-validation (a stricter, leakage-free setup), "
        "so its number isn't a perfectly like-for-like comparison with the others."
    )

    st.divider()

    # --- What each model looks at ---
    st.subheader("What each model pays attention to")

    import altair as alt

    def importance_chart(names, values, value_label, signed=False):
        """Horizontal bar chart; y-axis order is pinned to the input order."""
        d = pd.DataFrame({"gene": list(names), "val": list(values)})
        if signed:
            d["dir"] = np.where(d["val"] > 0, "toward metastatic", "toward primary")
            color = alt.Color(
                "dir:N",
                scale=alt.Scale(
                    domain=["toward metastatic", "toward primary"],
                    range=["#c0392b", "#2c6fbb"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            )
        else:
            color = alt.value("#2c6fbb")
        # ~28px per row so every label has room; labelLimit stops truncation
        chart_height = max(300, len(d) * 28)
        return (
            alt.Chart(d)
            .mark_bar()
            .encode(
                x=alt.X("val:Q", title=value_label),
                y=alt.Y(
                    "gene:N",
                    sort=list(names),
                    title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=200),
                ),
                color=color,
                tooltip=["gene", alt.Tooltip("val:Q", format=".3f")],
            )
            .properties(height=chart_height)
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM"]
    )

    with tab1:
        st.write(
            "Logistic regression assigns each gene a **signed weight**. Positive "
            "(pushes toward *metastatic*), negative (pushes toward *primary*)."
        )
        lr_w = logreg.coef_[0]
        order = np.argsort(np.abs(lr_w))[::-1][:15]
        st.altair_chart(
            importance_chart(
                [TOP100[i] for i in order],
                [float(lr_w[i]) for i in order],
                "signed weight",
                signed=True,
            ),
            use_container_width=True,
        )
        st.caption("Top gene: C7 (positive → metastatic). Many keratin genes push toward primary.")

    with tab2:
        st.write(
            "The decision tree uses only a handful of genes — the ones that split "
            "the data most cleanly. Everything in the app's diagnosis page runs on these."
        )
        dt_imp = tree.feature_importances_
        used = [(TOP100[i], float(dt_imp[i])) for i in range(len(TOP100)) if dt_imp[i] > 0]
        used.sort(key=lambda t: t[1], reverse=True)
        st.altair_chart(
            importance_chart([g for g, _ in used], [v for _, v in used], "gini importance"),
            use_container_width=True,
        )
        st.caption("S100A7 dominates, doing most of the splitting work.")

    with tab3:
        st.write(
            "Random forest builds 500 trees and reports how often each gene was "
            "useful for splitting — its **importance**."
        )
        rf_imp = rf.feature_importances_
        order = np.argsort(rf_imp)[::-1][:15]
        st.altair_chart(
            importance_chart(
                [TOP100[i] for i in order],
                [float(rf_imp[i]) for i in order],
                "gini importance",
            ),
            use_container_width=True,
        )
        st.caption("C7 again ranks highest, with S100A7 and KRT17 close behind.")

    with tab4:
        st.write(
            "XGBoost builds trees one after another, each correcting the last one's "
            "mistakes (boosting). It reports how much each gene improves the model — "
            "its **importance**."
        )
        xgb_imp = xgb.feature_importances_
        order = np.argsort(xgb_imp)[::-1][:15]
        st.altair_chart(
            importance_chart(
                [TOP100[i] for i in order],
                [float(xgb_imp[i]) for i in order],
                "importance (gain)",
            ),
            use_container_width=True,
        )
        st.caption(
            f"XGBoost reaches the highest accuracy here ({ACC['xgb']*100:.1f}%), "
            "leaning heavily on a few top genes rather than spreading attention evenly."
        )

    with tab5:
        st.write(
            "SVM (linear) draws the boundary that best separates the two groups, and "
            "like logistic regression it gives each gene a **signed weight**. Positive "
            "(pushes toward *metastatic*), negative (pushes toward *primary*)."
        )
        svm_w = meta.get("svm_weights", {})
        items = sorted(svm_w.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
        st.altair_chart(
            importance_chart(
                [g for g, _ in items],
                [w for _, w in items],
                "signed weight",
                signed=True,
            ),
            use_container_width=True,
        )
        st.caption(
            "C7 tops SVM's list too — the same gene logistic regression, random forest, "
            "and XGBoost all lean on. This model was trained with a stricter, "
            "leakage-free 5-fold setup."
        )

    st.divider()

    # --- The shared insight ---
    st.subheader("Where they agree — and where they don't")
    st.write(
        "Despite working differently, all five models keep surfacing the same few "
        "genes — especially **C7** and **S100A7**. When independent methods agree, "
        "it's a stronger signal that those genes really do carry information about "
        "whether melanoma has spread."
    )

    st.markdown("**What these genes are:**")
    with st.expander("C7 — complement component 7"):
        st.write(
            "A protein of the complement system, part of the body's innate immune "
            "defense that helps coordinate immune responses. Its link to cancer is "
            "complex: in some cancers lower C7 tracks with worse outcomes (a possible "
            "tumor-suppressor role), while in others higher C7 tracks with spread. "
            "That context-dependence is part of why it stands out here."
        )
    with st.expander("S100A7 — psoriasin"):
        st.write(
            "A calcium-binding protein made by skin keratinocytes, first found in "
            "psoriasis. It's tied to inflammation, attracting immune cells, and — in "
            "several epithelial skin tumors — is turned up as tumors progress."
        )
    with st.expander("KRT17 — keratin 17"):
        st.write(
            "A structural filament protein in skin cells that also helps regulate cell "
            "growth, immune response, and differentiation. In many cancers higher "
            "KRT17 tracks with more proliferation and invasion, though its role can "
            "flip depending on the tumor type."
        )
    st.caption(
        "These roles come from cancer research broadly, not proof about melanoma "
        "specifically — the app shows correlation the model found, not established cause."
    )

    st.markdown("**But they don't fully agree — look at KRT17:**")
    st.write(
        "KRT17 is a good example of why comparing models matters. Logistic regression "
        "ranks it near the top (as a *primary* signal), and random forest rates it "
        "highly too — but the single decision tree barely leans on it. Why the gap?"
    )
    st.markdown(
        "- **Different definitions of 'important.'** Each model measures importance its "
        "own way — a signed weight, a vote count across 500 trees, or how cleanly one "
        "tree splits the data.\n"
        "- **The tree is shallow (depth 3).** KRT17 only appears in one low branch, so "
        "its share looks small — while the forest's 500 trees give it many more chances "
        "to shine.\n"
        "- **Genes overlap.** KRT17 behaves like other keratin genes, so a single tree "
        "can lean on just one of them, while the forest spreads the credit around."
    )
    st.info(
        "Takeaway: a gene isn't simply 'important' or 'not.' It depends on how you look "
        "— which is exactly why using several models tells a richer story than one.",
        icon="💡",
    )
