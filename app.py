"""
AI-Powered Home Maintenance Spotter
===================================
A RAG-based application that analyzes photos of house problems and provides
actionable maintenance advice using home inspection knowledge base.
"""

import base64
import json
from pathlib import Path

import streamlit as st
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"
CHROMA_PERSIST_DIR = "./chroma_db_json"

EMBEDDING_MODEL = "text-embedding-3-small"
VISION_MODEL = "gpt-4.1"
TOP_K_RETRIEVAL = 3

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Home Maintenance Spotter",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom styling for a modern, clean look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4a4a6a;
        margin-bottom: 2rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# LOAD KNOWLEDGE BASE FROM JSON (Python native, no jq/LangChain JSONLoader)
# -----------------------------------------------------------------------------
def load_knowledge_base_json() -> list[dict]:
    """Load knowledge_base.json using Python's native json module."""
    path = KNOWLEDGE_BASE_PATH.resolve()
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError("knowledge_base.json is empty")
    return json.loads(raw)


def create_documents_from_json(data: list[dict]) -> list[Document]:
    """
    Create LangChain Document objects from the JSON array.
    page_content = issue + description (for embedding context)
    metadata = category, severity, action
    """
    documents = []
    for item in data:
        page_content = f"{item.get('issue', '')} {item.get('description', '')}".strip()
        metadata = {
            "category": item.get("category", ""),
            "severity": item.get("severity", ""),
            "action": item.get("action", ""),
            "id": item.get("id", ""),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


# -----------------------------------------------------------------------------
# VECTOR DATABASE INITIALIZATION (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def init_vector_store(api_key: str):
    """
    Load knowledge_base.json, create Documents, embed with text-embedding-3-small,
    and store in Chroma at ./chroma_db_json. Cached so it runs only once.
    
    RAG Flow Step 0: Embed documents and store in vector DB.
    """
    data = load_knowledge_base_json()
    documents = create_documents_from_json(data)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="home_inspection_rules",
        persist_directory=CHROMA_PERSIST_DIR,
    )

    return vector_store


# -----------------------------------------------------------------------------
# OPENAI VISION: Get image description (Step 1 of RAG pipeline)
# -----------------------------------------------------------------------------
def _image_url(image_base64: str, mime_type: str = "image/jpeg") -> str:
    """Build data URL for OpenAI vision API."""
    return f"data:{mime_type};base64,{image_base64}"


def get_image_description(
    client: OpenAI, images: list[tuple[str, str]]
) -> tuple[str, dict]:
    """
    Send one or more images to GPT-4.1 Vision for description.
    images: list of (base64, mime_type) tuples.
    Returns (description, usage_dict).
    """
    content = [
        {
            "type": "text",
            "text": (
                "You are a home inspection assistant. Look at the photo(s) of house problem(s) "
                "and write a brief description (in Greek) of what you see in each. "
                "Focus on: type of damage, location, visible severity. Be concise and factual. "
                "If multiple photos: describe all problems found across the images."
            ),
        },
    ]
    for base64_str, mime in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_url(base64_str, mime)},
        })
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=300,
    )
    u = response.usage
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    } if u else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return response.choices[0].message.content.strip(), usage


# -----------------------------------------------------------------------------
# RAG RETRIEVAL: Get relevant guidelines (Step 2 of RAG pipeline)
# -----------------------------------------------------------------------------
def retrieve_guidelines(vector_store, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[Document]:
    """
    Query the vector database with the image description to retrieve the most
    relevant home inspection guidelines. Returns full Document objects (with metadata).
    """
    return vector_store.similarity_search(query, k=top_k)


# -----------------------------------------------------------------------------
# FINAL ASSESSMENT: Vision + RAG combined (Step 3 of RAG pipeline)
# -----------------------------------------------------------------------------
def get_final_assessment(
    client: OpenAI,
    images: list[tuple[str, str]],
    retrieved_docs: list[Document],
) -> tuple[str, dict]:
    """
    Send images and retrieved guidelines to GPT-4.1 for assessment.
    images: list of (base64, mime_type) tuples.
    """
    # Format each guideline with page_content AND metadata (severity, action)
    guideline_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        severity = doc.metadata.get("severity", "")
        action = doc.metadata.get("action", "")
        content = doc.page_content
        block = f"""Guideline {i}:
- Description: {content}
- Severity (predefined): {severity}
- Action (predefined): {action}"""
        guideline_blocks.append(block)

    guidelines_text = "\n\n".join(guideline_blocks)

    system_prompt = """Είσαι ειδικός ελεγκτής σπιτιών. Ανάλυσε τη/τις φωτογραφία/ίες προβλημάτων 
και δώσε δομημένη αξιολόγηση. Η συμβουλή σου ΠΡΕΠΕΙ να βασίζεται ΑΠΟΚΛΕΙΣΤΙΚΑ στα οδηγά που δίνονται. 
ΜΗΝ επινοείς συμβουλές εκτός των οδηγών.

ΚΡΙΣΙΜΟ: Βάσισε τη συμβουλή σου στην προκαθορισμένη "Action" κάθε οδηγού.

Μορφή απάντησης (στα Ελληνικά):
1. **Πρόβλημα**: Σύντομη περιγραφή αυτού που βλέπεις στη/τις φωτογραφία/ίες
2. **Σοβαρότητα**: Χρησιμοποίησε τη προκαθορισμένη Severity (Χαμηλή, Μέτρια, Υψηλή ή Κρίσιμη)
3. **Πρακτικές συμβουλές**: Βήμα-βήμα βασισμένα στην Action και τους οδηγούς
4. **Πότε να καλέσετε επαγγελματία**: Συνθήκες που απαιτούν επαγγελματία (από τους οδηγούς μόνο)"""

    user_content = f"""Οδηγοί ελεγκτικής σπιτιού (με προκαθορισμένη Severity και Action):

{guidelines_text}

---

Ανάλυσε τη/τις συνημμένη/νες φωτογραφία/ίες και δώσε την αξιολόγησή σου στα Ελληνικά. Χρησιμοποίησε τη Severity και Action παραπάνω."""

    user_content_parts = [{"type": "text", "text": user_content}]
    for base64_str, mime in images:
        user_content_parts.append({
            "type": "image_url",
            "image_url": {"url": _image_url(base64_str, mime)},
        })
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_parts},
        ],
        max_tokens=1200,
    )
    u = response.usage
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    } if u else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return response.choices[0].message.content.strip(), usage


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    # Header
    st.markdown('<p class="main-header">🏠 AI-Powered Home Maintenance Spotter</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ανεβάστε μία ή περισσότερες φωτογραφίες προβλημάτων σπιτιού '
        '(συρίζοντας σωλήνα, ρωγμές, μούχλα κ.λπ.) και λάβετε συμβουλές βασισμένες σε οδηγούς ελεγκτικής.</p>',
        unsafe_allow_html=True,
    )

    # Sidebar: API Key
    with st.sidebar:
        st.header("🔐 Ρυθμίσεις API")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Απαιτείται για την ανάλυση εικόνων. Δεν αποθηκεύεται.",
        )
        st.divider()
        st.caption("Το API key χρησιμοποιείται μόνο για αυτή τη συνεδρία.")

    # Validate API key
    if not api_key or not api_key.strip().startswith("sk-"):
        st.warning(
            "⚠️ Εισάγετε το OpenAI API key στη στήλη για να χρησιμοποιήσετε την εφαρμογή. "
            "[platform.openai.com](https://platform.openai.com/api-keys)"
        )
        st.info(
            "Μετά μπορείτε να ανεβάσετε φωτογραφία/ίες και να πατήσετε **Ανάλυση**."
        )
        return
    
    # Initialize vector store (cached)
    try:
        vector_store = init_vector_store(api_key)
    except FileNotFoundError:
        st.error(
            f"Knowledge base file not found: `{KNOWLEDGE_BASE_PATH.resolve()}`. "
            "Please ensure `knowledge_base.json` exists in the app directory."
        )
        return
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Invalid knowledge base: {e}")
        return
    except Exception as e:
        st.error(f"Failed to initialize knowledge base: {e}")
        return
    
    # Image upload (multiple files)
    uploaded_files = st.file_uploader(
        "Ανεβάστε φωτογραφία/ίες προβλημάτων",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Μπορείτε να επιλέξετε πολλές φωτογραφίες. JPG, PNG.",
    )

    if uploaded_files:
        # Display all uploaded images
        for i, uf in enumerate(uploaded_files):
            st.image(uf, caption=f"Φωτογραφία {i + 1}: {uf.name}")
        # Re-read after display (Streamlit consumes the file buffer)
        images_data = []
        for uf in uploaded_files:
            uf.seek(0)
            img_bytes = uf.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            mt = "image/png" if uf.name.lower().endswith(".png") else "image/jpeg"
            images_data.append((img_b64, mt))

        if st.button("🔍 Ανάλυση φωτογραφιών", type="primary"):
            client = OpenAI(api_key=api_key)

            with st.spinner("Ανάλυση εικόνων και σύγκριση με τη βάση γνώσεων..."):
                try:
                    description, usage1 = get_image_description(client, images_data)
                    retrieved_docs = retrieve_guidelines(vector_store, description)
                    assessment, usage2 = get_final_assessment(
                        client, images_data, retrieved_docs
                    )

                    total_tokens = (
                        usage1.get("total_tokens", 0) + usage2.get("total_tokens", 0)
                    )
                    
                except Exception as e:
                    st.error(f"Αποτυχία ανάλυσης: {e}")
                    return

            st.success("Η ανάλυση ολοκληρώθηκε!")

            # Parse severity (English or Greek)
            severity_lower = "low"
            for line in assessment.split("\n"):
                low = line.lower()
                if "σοβαρότητα" in low or "severity" in low or "**" in line:
                    s = line.split(":")[-1].strip().lower() if ":" in line else low
                    if "κρίσιμ" in s or "critical" in s:
                        severity_lower = "critical"
                    elif "υψηλή" in s or "high" in s:
                        severity_lower = "high"
                    elif "μέτρι" in s or "medium" in s:
                        severity_lower = "medium"
                    break

            if severity_lower == "critical":
                st.error("🔴 **Κρίσιμη σοβαρότητα** — Χρειάζεται άμεση αντιμετώπιση.")
            elif severity_lower == "high":
                st.error("🔴 **Υψηλή σοβαρότητα** — Αντιμετωπίστε σύντομα.")
            elif severity_lower == "medium":
                st.warning("🟡 **Μέτρια σοβαρότητα** — Προγραμματίστε την επιδιόρθωση.")
            else:
                st.info("🟢 **Χαμηλή σοβαρότητα** — Παρακολουθήστε το πρόβλημα.")

            st.markdown("### 📋 Αξιολόγηση")
            st.markdown(assessment)
            st.caption(f"📊 Συνολικά tokens: **{total_tokens}**")

    else:
        st.info("👆 Ανεβάστε μία ή περισσότερες φωτογραφίες JPG/PNG.")


if __name__ == "__main__":
    main()
