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
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return nlp, model

nlp, model = load_models()

# -----------------------------
# Utility: Normalize Entities
# -----------------------------
def normalize(text):
    return text.lower().replace("the ", "").strip()

# -----------------------------
# Step 1: Named Entity Recognition
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

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
def link_domains(triples_df, threshold=0.6):
    # Create full semantic phrases from triples
    triples_df["sentence"] = triples_df.apply(
        lambda row: f"{row['Entity1']} {row['Relation']} {row['Entity2']}", axis=1
    )
    sentences = triples_df["sentence"].drop_duplicates().tolist()

    if len(sentences) < 2:
        return []

    embeddings = model.encode(sentences, convert_to_tensor=True)
    linked = []

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            linked.append((sentences[i], sentences[j], round(score, 3)))

    # Show top 10 links
    top_links = sorted(linked, key=lambda x: x[2], reverse=True)[:10]
    return [link for link in top_links if link[2] > threshold]

# -----------------------------
# Step 4: Visualization & Analytics
# -----------------------------
def visualize_knowledge_graph(triples_df, highlight_nodes=None):
    G = nx.DiGraph()
    for _, row in triples_df.iterrows():
        G.add_node(row["Entity1"])
        G.add_node(row["Entity2"])
        G.add_edge(row["Entity1"], row["Entity2"], label=row["Relation"])

    st.write("### Graph Analytics")
    if len(G.nodes) > 0:
        degree_centrality = nx.degree_centrality(G)
        top_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        st.write("#### Top 5 Central Nodes (by Degree Centrality)")
        for node, score in top_central_nodes:
            st.write(f"- **{node}** → {score:.3f}")

        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))
            st.write(f"#### Detected {len(communities)} Communities:")
            for i, community in enumerate(communities):
                st.write(f"**Community {i+1}:** {list(community)}")
        except Exception as e:
            st.warning(f"Community detection skipped: {e}")
    else:
        st.warning("No nodes available for centrality or community analysis.")

    net = Network(height="600px", width="100%", directed=True)
    for node in G.nodes:
        net.add_node(node, label=node, color="red" if highlight_nodes and node in highlight_nodes else None)
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
st.title("Semantic Knowledge Graph Generator (Enhanced Version)")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
triples = []

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    if "sentence" not in df.columns:
        st.error("File must contain a column named 'sentence'")
    else:
        st.info("Using column: **sentence**")
        for text in df["sentence"].dropna():
            triples.extend(extract_relations(text))

        triples_df = pd.DataFrame(triples, columns=["Entity1", "Relation", "Entity2"])
        triples_df.to_csv("triples_output.csv", index=False)
        st.success("Triples extracted and saved to triples_output.csv")

        st.write("### Extracted Triples")
        st.dataframe(triples_df, use_container_width=True, height=400)

        # -----------------------------
        # Semantic Search
        # -----------------------------
        st.write("### Search the Knowledge Graph")
        query = st.text_input("Enter your search query:", "")
        highlight_nodes = None

        if st.button("Search") and query:
            all_nodes = list(set(triples_df["Entity1"].tolist() + triples_df["Entity2"].tolist()))
            node_embeddings = model.encode(all_nodes, convert_to_tensor=True)
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
            results = sorted(zip(all_nodes, cosine_scores), key=lambda x: x[1], reverse=True)[:5]

            st.write("### Top Matches:")
            for node, score in results:
                st.write(f"- {node} ({score:.4f})")
            highlight_nodes = [r[0] for r in results]

        # -----------------------------
        # Query Answering
        # -----------------------------
        st.write("### Query Answering")
        question = st.text_input("Ask a question (e.g., 'What is the capital of France?')")

        if st.button("Get Answer") and question:
            q_doc = nlp(question)
            q_ents = [ent.text for ent in q_doc.ents]
            q_tokens = [token.text for token in q_doc if token.pos_ in ["NOUN", "PROPN"]]

            found = []
            for _, row in triples_df.iterrows():
                if any(ent.lower() in row["Entity1"].lower() or ent.lower() in row["Entity2"].lower()
                       for ent in q_ents + q_tokens):
                    found.append(row)

            if found:
                st.write("### Possible Answers:")
                seen = set()
                for _, row in pd.DataFrame(found).iterrows():
                    sentence = f"{row['Entity1']} {row['Relation']} {row['Entity2']}"
                    if sentence not in seen:
                        st.write(f"- **{sentence}**")
                        seen.add(sentence)
            else:
                st.write("No direct match found. Try rephrasing your question.")

        # -----------------------------
        # Domain Linking
        # -----------------------------
        st.write("### Domain Linking (Similar Entities)")
        domain_links = link_domains(triples_df)
        if domain_links:
            st.dataframe(pd.DataFrame(domain_links, columns=["Entity1", "Entity2", "Similarity"]))
        else:
            st.write("No strong semantic links found.")

        # -----------------------------
        # Visualize Graph
        # -----------------------------
        st.write("### Semantic Knowledge Graph Visualization")
        visualize_knowledge_graph(triples_df, highlight_nodes=highlight_nodes)
