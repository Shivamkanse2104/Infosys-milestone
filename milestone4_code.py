import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Named Entity Recognition
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Step 2: Relation Extraction
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

# Step 3: Semantic Map Visualization
def visualize_knowledge_graph(triples_df,  highlight_nodes=None):
    G = nx.DiGraph()

    # Add nodes and edges
    for _, row in triples_df.iterrows():
        entity1 = row["Entity1"]
        relation = row["Relation"]
        entity2 = row["Entity2"]
        G.add_node(entity1)
        G.add_node(entity2)
        G.add_edge(entity1, entity2, label=relation)

    # Use PyVis for interactive visualization
    net = Network(height="600px", width="100%", directed=True)
    for node in G.nodes:
        if highlight_nodes and node in highlight_nodes:
            net.add_node(node, color="red", size=30, label=node)
        else:
            net.add_node(node, label=node)
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], label=edge[2]["label"])

    # Customize edge labels
    for edge in net.edges:
        edge["title"] = edge["label"]

    # Save and display in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name)
        components.html(open(tmp_file.name, "r").read(), height=600)
    


# Streamlit App
st.title("Semantic Knowledge Graph from 'sentence' Column")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

triples = []

if uploaded_file:
    # Detect file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Use the 'sentence' column directly
    if "sentence" not in df.columns:
        st.error("File must contain a column named 'sentence'")
    else:
        st.info("Using column: **sentence**")

        # Loop through ALL rows in 'sentence'
        for text in df["sentence"].dropna():
            entities = extract_entities(text)
            relations = extract_relations(text)
            triples.extend(relations)

        # Save triples to CSV
        triples_df = pd.DataFrame(triples, columns=["Entity1", "Relation", "Entity2"])
        triples_df.to_csv("triples_output.csv", index=False)
        st.success("Triples extracted and saved to triples_output.csv")

        # Show ALL triples in scrollable table
        st.write("### Extracted Triples")
        st.dataframe(triples_df, use_container_width=True, height=600)

        # Search Box
        st.write("### Search the Knowledge Graph")
        query = st.text_input("Enter your search query:", "")
        highlight_nodes = None

        if st.button("Search") and query:
            all_nodes = list(set(triples_df["Entity1"].tolist() + triples_df["Entity2"].tolist()))
            node_embeddings = model.encode(all_nodes, convert_to_tensor=True)
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
            top_k = 5
            results = sorted(zip(all_nodes, cosine_scores), key=lambda x: x[1], reverse=True)[:top_k]

            st.write("### Top Matches:")
            for node, score in results:
                st.write(f"{node} ({score:.4f})")

            highlight_nodes = [r[0] for r in results]
    st.write("### Semantic Map (Knowledge Graph)")
    visualize_knowledge_graph(triples_df,highlight_nodes=highlight_nodes)
