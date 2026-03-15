# 🏠 HouseMaxing
### AI-Powered Home Inspector & Airbnb Room Stager

> Ανέβασε μια φωτογραφία. Πάρε επαγγελματική αξιολόγηση σε δευτερόλεπτα.

HouseMaxing χρησιμοποιεί **GPT-4.1 Vision** και μια **RAG pipeline** με 80 κανόνες ειδικών για να αναλύει φωτογραφίες χώρων και να παράγει αναφορές με βαθμολογία, ευρήματα και σχέδιο δράσης — στα ελληνικά.

---

## 📁 Δομή Αρχείων

```
HouseMaxing/
├── app.py                      # Κύρια Streamlit εφαρμογή
├── Ingest.py                   # Embedding script (τρέχει αυτόματα)
├── knowledge_base_airbnb.json  # 80 κανόνες ειδικών σε 8 κατηγορίες
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── entrypoint.sh               # Auto-ingest + Streamlit launcher
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
└── README.md                   # Αυτό το αρχείο
```

---

## ⚡ Εκκίνηση — 2 Εντολές

### 🐳 Με Docker (Μοναδική επιλογή για αξιολόγηση)

```bash
# 1. Build
docker build -t housemaxing .

# 2. Run — αντικατέστησε το sk-... με το πραγματικό σου key
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... housemaxing

# 3. Άνοιξε τον browser:
# http://localhost:8501
```

> ✅ **Δεν χρειάζεται τίποτα άλλο.** Το `entrypoint.sh` τρέχει αυτόματα το embedding (~30 δευτ.) και ξεκινά την εφαρμογή.  
> Το API key περνάει μέσω του `-e` flag — δεν αποθηκεύεται πουθενά στον κώδικα.

---

### 💻 Χωρίς Docker (Εναλλακτικά)

```bash
# 1. Εγκατάσταση
pip install -r requirements.txt

# 2. Εκκίνηση
python3 -m streamlit run app.py
```

Στη web εφαρμογή:
1. Εισάγαγε το **OpenAI API key** σου (ξεκινά με `sk-`)
2. Πάτα **✦ Συνέχεια**
3. Η εφαρμογή κάνει αυτόματα το embedding (~30 δευτ.)
4. Ανέβασε φωτογραφία και πάτα **Ανάλυση**

---

## 🧠 Αρχιτεκτονική — Πώς Λειτουργεί

```
Φωτογραφία
    │
    ▼
GPT-4.1 Vision
Αναλύει την εικόνα σε 9 κατηγορίες
(φωτισμός, έπιπλα, ασφάλεια κ.λπ.)
    │
    ▼
ChromaDB + LangChain (RAG)
Semantic search στους 80 κανόνες ειδικών
Επιστρέφει τους 8 πιο σχετικούς
    │
    ▼
GPT-4.1 Report Generator
Παράγει structured JSON αναφορά
βασισμένη αποκλειστικά στους retrieved κανόνες
    │
    ▼
Streamlit UI
Βαθμολογία /10 · Ευρήματα · Συμβουλές
```

---

## 🗂️ Κατηγορίες Κανόνων (80 κανόνες)

| # | Κατηγορία | Κανόνες |
|---|-----------|---------|
| 1 | 🔆 Φωτισμός & Φωτογραφία | 10 |
| 2 | 🛏️ Staging Υπνοδωματίου | 10 |
| 3 | 🚿 Staging Μπάνιου | 10 |
| 4 | 🛋️ Χώροι Διαβίωσης | 10 |
| 5 | 🍳 Staging Κουζίνας | 10 |
| 6 | 🏡 Εξωτερικός Χώρος & Curb Appeal | 10 |
| 7 | 🔒 Ασφάλεια & Συμμόρφωση | 10 |
| 8 | 🎯 Εμπειρία Επισκέπτη & Ανέσεις | 10 |

---

## 🔍 Τρόποι Ανάλυσης

| Τρόπος | Χρήση |
|--------|-------|
| **Staging Airbnb** | Ποιότητα design για βραχυχρόνιες μισθώσεις |
| **Έλεγχος Συντήρησης** | Εντοπισμός επισκευών & θεμάτων ασφάλειας |
| **Πλήρης Επιθεώρηση** | Πλήρης αισθητική & δομική αξιολόγηση |
| **Εξωτερικός Έλεγχος** | Curb appeal, στέγη, θεμέλια, διαμόρφωση |

---

## 📊 Δομή Αναφοράς

Κάθε ανάλυση παράγει:

- **Βαθμολογία** `/10` με animated ring indicator
- **Σύνοψη** — 2-3 προτάσεις executive summary
- **Τι Λειτουργεί Καλά** — θετικά στοιχεία του χώρου
- **Ευρήματα & Σχέδιο Δράσης** — severity badges (Critical / High / Medium / Low) με ακριβείς οδηγίες
- **Επαγγελματικές Συμβουλές** — photo tips και actionable βελτιώσεις
- **Κανόνες που Αντιστοιχήθηκαν** — πλήρης λίστα με rule IDs
- **Token Counter** — χρήση tokens και εκτιμώμενο κόστος ανά ανάλυση

---

## 🛠️ Tech Stack

| Στοιχείο | Τεχνολογία |
|----------|-----------|
| Frontend | Python + Streamlit |
| Vision & LLM | OpenAI GPT-4.1 |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | ChromaDB (local persistent) |
| RAG Framework | LangChain |
| Knowledge Base | JSON (80 expert rules) |
| Containerization | Docker (python:3.11-slim) |

---

## 💰 Κόστος Tokens

| Κατηγορία | Τιμή |
|-----------|------|
| Input tokens (GPT-4.1) | $2 / 1M tokens |
| Output tokens (GPT-4.1) | $8 / 1M tokens |
| Embeddings | $0.02 / 1M tokens |

> Μια τυπική ανάλυση μιας φωτογραφίας κοστίζει περίπου **$0.01–$0.03**.

---

## 📋 Requirements

```
streamlit>=1.35.0
openai>=1.30.0
langchain>=0.2.0
langchain-openai>=0.1.8
langchain-chroma>=0.1.1
chromadb>=0.5.0
Pillow>=10.0.0
```

> **Python 3.11** συνιστάται (χρησιμοποιείται και στο Docker image).

---

*Built for Netcompany Hackathon Thessaloniki 2026 — HouseMaxing*
*© 2026 NETTER. All Rights Reserved. Unauthorized copying or use of this code is strictly prohibited.*
