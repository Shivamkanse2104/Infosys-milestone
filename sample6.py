import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
from sentence_transformers import SentenceTransformer, util
import torch

# -----------------------------
# Model Loading (Safe CPU mode)
# -----------------------------
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    # Force CPU mode to avoid meta tensor error
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return nlp, model

nlp, model = load_models()

# -----------------------------
# Step 1: Named Entity Recognition
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# -----------------------------
# Step 2: Relation Extraction
# -----------------------------
def extract_relations(text):
    doc = nlp(text)
    relations = []
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = [w.text for w in token.lefts if w.dep_ in ["nsubj", "nsubjpass"]]
            obj = [w.text for w in token.rights if w.dep_ in ["dobj", "attr", "dative", "oprd"]]
            if subject and obj:
                relations.append((subject[0], token.text, obj[0]))
    return relations

# -----------------------------
# Step 3: Domain Linking (Semantic Similarity)
# -----------------------------
def link_domains(triples_df, threshold=0.75):
    entities = list(set(triples_df["Entity1"].tolist() + triples_df["Entity2"].tolist()))
    if len(entities) < 2:
        return []
    embeddings = model.encode(entities, convert_to_tensor=True)
    linked = []
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i < j:
                score = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                if score > threshold:
                    linked.append((e1, e2, round(score, 3)))
    return linked

# -----------------------------
# Step 4: Visualization & Analytics
# -----------------------------
def visualize_knowledge_graph(triples_df, highlight_nodes=None):
    G = nx.DiGraph()

    # Add nodes and edges
    for _, row in triples_df.iterrows():
        entity1 = row["Entity1"]
        relation = row["Relation"]
        entity2 = row["Entity2"]
        G.add_node(entity1)
        G.add_node(entity2)
        G.add_edge(entity1, entity2, label=relation)

    # --- Graph Analytics ---
    st.write("### 📊 Graph Analytics")

    # Centrality
    if len(G.nodes) > 0:
        degree_centrality = nx.degree_centrality(G)
        top_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        st.write("#### 🔹 Top 5 Central Nodes (by Degree Centrality)")
        for node, score in top_central_nodes:
            st.write(f"- **{node}** → {score:.3f}")

        # Community Detection
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))
            st.write(f"#### 🧩 Detected {len(communities)} Communities:")
            for i, community in enumerate(communities):
                st.write(f"**Community {i+1}:** {list(community)}")
        except Exception as e:
            st.warning(f"Community detection skipped: {e}")
    else:
        st.warning("No nodes available for centrality or community analysis.")

    # --- Visualization (PyVis) ---
    net = Network(height="600px", width="100%", directed=True)
    for node in G.nodes:
        if highlight_nodes and node in highlight_nodes:
            net.add_node(node, color="red", size=30, label=node)
        else:
            net.add_node(node, label=node)

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], label=edge[2]["label"])

    for edge in net.edges:
        edge["title"] = edge["label"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name)
        components.html(open(tmp_file.name, "r").read(), height=600)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🧠 Semantic Knowledge Graph Generator (Enhanced Version)")

uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv", "xlsx"])

triples = []

if uploaded_file:
    # Detect file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "sentence" not in df.columns:
        st.error("File must contain a column named 'sentence'")
    else:
        st.info("Using column: **sentence**")

        # Extract triples
        for text in df["sentence"].dropna():
            relations = extract_relations(text)
            triples.extend(relations)

        # Store as DataFrame
        triples_df = pd.DataFrame(triples, columns=["Entity1", "Relation", "Entity2"])
        triples_df.to_csv("triples_output.csv", index=False)
        st.success("✅ Triples extracted and saved to triples_output.csv")

        st.write("### 📋 Extracted Triples")
        st.dataframe(triples_df, use_container_width=True, height=400)

        # -----------------------------
        # Semantic Search
        # -----------------------------
        st.write("### 🔍 Search the Knowledge Graph")
        query = st.text_input("Enter your search query:", "")
        highlight_nodes = None

        if st.button("Search") and query:
            all_nodes = list(set(triples_df["Entity1"].tolist() + triples_df["Entity2"].tolist()))
            node_embeddings = model.encode(all_nodes, convert_to_tensor=True)
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
            top_k = 5
            results = sorted(zip(all_nodes, cosine_scores), key=lambda x: x[1], reverse=True)[:top_k]

            st.write("### 🧾 Top Matches:")
            for node, score in results:
                st.write(f"- {node} ({score:.4f})")

            highlight_nodes = [r[0] for r in results]

        # -----------------------------
        # Query Answering
        # -----------------------------
        st.write("### ❓ Query Answering")
        question = st.text_input("Ask a question (e.g., 'What is the capital of France?')")

        if st.button("Get Answer") and question:
            q_doc = nlp(question)
            q_ents = [ent.text for ent in q_doc.ents]
            q_tokens = [token.text for token in q_doc if token.pos_ in ["NOUN", "PROPN"]]

            found = []
            for _, row in triples_df.iterrows():
                if any(ent.lower() in row["Entity1"].lower() or ent.lower() in row["Entity2"].lower() for ent in q_ents + q_tokens):
                    found.append(row)

            if found:
                st.write("### ✅ Possible Answers:")
                ans_df = pd.DataFrame(found)
                st.dataframe(ans_df)
            else:
                st.write("No direct match found. Try rephrasing your question.")

        # -----------------------------
        # Domain Linking
        # -----------------------------
        st.write("### 🔗 Domain Linking (Similar Entities)")
        domain_links = link_domains(triples_df)
        if domain_links:
            st.dataframe(pd.DataFrame(domain_links, columns=["Entity1", "Entity2", "Similarity"]))
        else:
            st.write("No strong semantic links found.")

        # -----------------------------
        # Visualize Graph
        # -----------------------------
        st.write("### 🌐 Semantic Knowledge Graph Visualization")
        visualize_knowledge_graph(triples_df, highlight_nodes=highlight_nodes)






import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
from sentence_transformers import SentenceTransformer, util
import torch
import community as community_louvain  # For community detection

# -------------------------------
# Prevent GPU/meta tensor errors
# -------------------------------
torch.set_default_device("cpu")

# -------------------------------
# Load Models
# -------------------------------
st.set_page_config(page_title="Semantic Knowledge Graph", layout="wide")
st.title("🧠 Semantic Knowledge Graph & Cross-Domain Linker")

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, model

nlp, model = load_models()

# -------------------------------
# Step 1: Named Entity Recognition
# -------------------------------
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# -------------------------------
# Step 2: Relation Extraction
# -------------------------------
def extract_relations(text):
    doc = nlp(text)
    relations = []
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = [w.text for w in token.lefts if w.dep_ in ["nsubj", "nsubjpass"]]
            obj = [w.text for w in token.rights if w.dep_ in ["dobj", "attr", "dative", "oprd"]]
            if subject and obj:
                relations.append((subject[0], token.text, obj[0]))
    return relations

# -------------------------------
# Step 3: Knowledge Graph Visualization
# -------------------------------
def visualize_knowledge_graph(triples_df, highlight_nodes=None):
    G = nx.DiGraph()

    # Add nodes & edges
    for _, row in triples_df.iterrows():
        entity1, relation, entity2 = row["Entity1"], row["Relation"], row["Entity2"]
        G.add_node(entity1)
        G.add_node(entity2)
        G.add_edge(entity1, entity2, label=relation)

    # Use PyVis for visualization
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    for node in G.nodes:
        if highlight_nodes and node in highlight_nodes:
            net.add_node(node, color="red", size=30, label=node)
        else:
            net.add_node(node, label=node)
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], label=edge[2]["label"])

    for edge in net.edges:
        edge["title"] = edge["label"]

    # Save and display graph
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name)
        components.html(open(tmp_file.name, "r").read(), height=600, scrolling=True)

# -------------------------------
# Upload CSV/Excel
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file (must contain a 'sentence' column)", type=["csv", "xlsx"])

triples = []

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "sentence" not in df.columns:
        st.error("File must contain a column named 'sentence'")
        st.stop()

    st.info("✅ Using column: **sentence**")

    # Extract triples
    for text in df["sentence"].dropna():
        relations = extract_relations(text)
        triples.extend(relations)

    # Store in DataFrame
    triples_df = pd.DataFrame(triples, columns=["Entity1", "Relation", "Entity2"])
    triples_df.to_csv("triples_output.csv", index=False)
    st.success("✅ Triples extracted and saved to triples_output.csv")

    # Display extracted triples
    st.write("### 📊 Extracted Triples")
    st.dataframe(triples_df, use_container_width=True, height=600)

    # -------------------------------
    # Centrality & Community Detection
    # -------------------------------
    st.write("### 🧩 Graph Analytics: Centrality & Community Detection")
    G = nx.from_pandas_edgelist(triples_df, "Entity1", "Entity2", edge_attr=True, create_using=nx.DiGraph())

    # Centrality Measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    centrality_df = pd.DataFrame({
        "Node": list(degree_centrality.keys()),
        "Degree Centrality": list(degree_centrality.values()),
        "Betweenness Centrality": list(betweenness_centrality.values())
    }).sort_values("Degree Centrality", ascending=False)

    st.dataframe(centrality_df, use_container_width=True, height=300)

    # Community Detection (Louvain)
    if len(G.nodes) > 0:
        partition = community_louvain.best_partition(G.to_undirected())
        st.write("Detected Communities:")
        communities = {}
        for node, comm in partition.items():
            communities.setdefault(comm, []).append(node)
        for comm_id, nodes in communities.items():
            st.write(f"**Community {comm_id + 1}:**", ", ".join(nodes[:10]), "...")
    else:
        st.warning("Graph is empty, cannot compute communities.")

    # -------------------------------
    # Query Answering and Cross-Domain Linking
    # -------------------------------
    st.write("### 🔍 Query Answering and Cross-Domain Linking")

    query = st.text_input("Enter your query (e.g., 'Einstein and Da Vinci'):")

    if st.button("Find Semantic Links"):
        if query.strip():
            all_nodes = list(set(triples_df["Entity1"].tolist() + triples_df["Entity2"].tolist()))
            node_embeddings = model.encode(all_nodes, convert_to_tensor=True)
            query_embedding = model.encode(query, convert_to_tensor=True)

            cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
            results = sorted(zip(all_nodes, cosine_scores), key=lambda x: x[1], reverse=True)

            threshold = 0.30  # Adjusted threshold
            filtered = [(node, score.item()) for node, score in results if score > threshold]

            if filtered:
                st.success(f"Found {len(filtered)} linked nodes (threshold={threshold})")
                for node, score in filtered:
                    st.write(f"• {node} (similarity={score:.3f})")

                highlight_nodes = [node for node, _ in filtered]
                st.write("### 🌐 Cross-Domain Semantic Map")
                visualize_knowledge_graph(triples_df, highlight_nodes=highlight_nodes)
            else:
                st.warning("No strong semantic links found. Try broader query or lower threshold.")
        else:
            st.info("Please enter a query to find cross-domain links.")

    # -------------------------------
    # Knowledge Graph Display
    # -------------------------------
    st.write("### 🌍 Full Knowledge Graph")
    visualize_knowledge_graph(triples_df)


