import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Directories and Model Config
PERSIST_DIR = "./chroma_data11"
COLLECTION_NAME = "cards_collection"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


@st.cache_resource
def get_sbert_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def get_unique_values_cached():
    """Cache unique values so they are computed only once."""
    collection = get_chroma_collection()
    return {
        'set_name': get_unique_values(collection, 'set_name'),
        'card_type': get_unique_values(collection, 'card_type'),
        'color': get_unique_values(collection, 'color'),
        'rarity': get_unique_values(collection, 'rarity'),
        'level': get_unique_values(collection, 'level'),
        'triggers': get_unique_values(collection, 'triggers'),
    }


def get_unique_values(collection, field_name):
    """Retrieve unique values for a given field."""
    unique_vals = set()
    limit, offset = 1000, 0
    while True:
        results = collection.get(include=["metadatas"], limit=limit, offset=offset)
        metadatas = results["metadatas"]
        if not metadatas:
            break
        for md in metadatas:
            val = md.get(field_name, "").strip()
            if val:
                unique_vals.add(val)
        offset += limit
    return sorted(list(unique_vals))


def build_filter_dict(set_name, card_type, color, rarity, level, triggers):
    """Constructs a filter dictionary for database queries."""
    f = {k: v for k, v in {
        "set_name": set_name,
        "card_type": card_type,
        "color": color,
        "rarity": rarity,
        "level": level,
        "triggers": triggers
    }.items() if v and v != "(No Filter)"}

    if len(f) > 1:
        return {"$and": [{k: v} for k, v in f.items()]}
    elif len(f) == 1:
        return f
    return {}


@st.cache_resource
def get_filtered_cards(filter_dict, limit):
    """Cache filtered card results to avoid redundant database queries."""
    collection = get_chroma_collection()
    return dumb_search(collection, filter_dict, limit)


def smart_search(collection, model, query_text, n_results, filter_dict):
    """Performs an AI-enhanced search using embeddings."""
    if not query_text.strip():
        return []

    query_embedding = model.encode(query_text).tolist()
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["metadatas", "distances", "documents"]
    }
    if filter_dict:
        query_kwargs["where"] = filter_dict

    results = collection.query(**query_kwargs)

    return [
        (results["ids"][0][i], results["documents"][0][i], results["metadatas"][0][i], results["distances"][0][i])
        for i in range(len(results["ids"][0]))
    ]


def dumb_search(collection, filter_dict, limit=500):
    """Fetches cards from the database using filters only."""
    query_kwargs = {
        "include": ["metadatas", "documents"],
        "limit": limit
    }
    if filter_dict:
        query_kwargs["where"] = filter_dict

    results = collection.get(**query_kwargs)

    return [
        (results["ids"][i], results["documents"][i], results["metadatas"][i], None)
        for i in range(len(results["ids"]))
    ]


def show_card_details(card_meta):
    """Displays details of a card."""
    if card_meta.get("image", ""):
        st.image(card_meta["image"], width=250)
    st.write(f"**Name**: {card_meta.get('name', '')}")
    st.write(f"**Code**: {card_meta.get('code', '')}")
    st.write(f"**Rarity**: {card_meta.get('rarity', '')}")
    st.write(f"**Expansion**: {card_meta.get('expansion', '')}")
    st.write(f"**Card Type**: {card_meta.get('card_type', '')}")
    st.write(f"**Color**: {card_meta.get('color', '')}")
    st.write(f"**Level**: {card_meta.get('level', '')}")
    st.write(f"**Cost**: {card_meta.get('cost', '')}")
    st.write(f"**Power**: {card_meta.get('power', '')}")
    st.write(f"**Triggers**: {card_meta.get('triggers', '')}")
    st.write(f"**Attributes**: {card_meta.get('attributes', '')}")
    st.write(f"**Abilities**: {card_meta.get('abilities', '')}")
    st.write(f"**Flavor Text**: {card_meta.get('flavor_text', '')}")
    st.write(f"**Set Name**: {card_meta.get('set_name', '')}")


def main():
    st.title("Weiss Card Search (scuff)")

    collection, sbert_model = get_chroma_collection(), get_sbert_model()
    unique_values = get_unique_values_cached()

    st.subheader("🔍 Filters")
    col1, col2 = st.columns(2)
    with col1:
        selected_set_name = st.selectbox("Set Name", ["(No Filter)"] + unique_values['set_name'])
        selected_color = st.selectbox("Color", ["(No Filter)"] + unique_values['color'])
        selected_level = st.selectbox("Level", ["(No Filter)"] + unique_values['level'])
    with col2:
        selected_card_type = st.selectbox("Card Type", ["(No Filter)"] + unique_values['card_type'])
        selected_rarity = st.selectbox("Rarity", ["(No Filter)"] + unique_values['rarity'])
        selected_triggers = st.selectbox("Triggers", ["(No Filter)"] + unique_values['triggers'])

    st.subheader("Smart Search (AI)")
    user_query = st.text_input("Enter your query text:", value="", help="ex. 'counter send opponent to memory'")
    n_results = st.slider("Number of results", min_value=1, max_value=30, value=10)

    if st.button("🔎 Run Smart Search", use_container_width=True):
        filter_dict = build_filter_dict(
            selected_set_name, selected_card_type, selected_color, selected_rarity, selected_level, selected_triggers
        )
        with st.spinner("Searching..."):
            results = smart_search(collection, sbert_model, user_query, n_results, filter_dict)
        st.success(f"Found {len(results)} result(s).")

        for card_id, card_doc, card_meta, distance in results:
            with st.expander(f"{card_meta.get('name', 'Unknown')} (Distance={round(distance, 3)})", expanded=False):
                show_card_details(card_meta)

    st.divider()

    st.subheader("Dumb Search (Filter-only mode)")
    dumb_limit = st.slider("Max number of results", min_value=10, max_value=500, value=50)

    if st.button("📄 Show Filtered Cards", use_container_width=True):
        filter_dict = build_filter_dict(
            selected_set_name, selected_card_type, selected_color, selected_rarity, selected_level, selected_triggers
        )
        with st.spinner("Fetching cards..."):
            results = get_filtered_cards(filter_dict, limit=dumb_limit)
        st.success(f"Showing {len(results)} card(s).")

        for card_id, card_doc, card_meta, _ in results:
            with st.expander(f"{card_meta.get('name', 'Unknown')}", expanded=False):
                show_card_details(card_meta)


if __name__ == "__main__":
    main()
