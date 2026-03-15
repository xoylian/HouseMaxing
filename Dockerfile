# Χρησιμοποιούμε επίσημο Linux image με Python 3.11 για απόλυτη σταθερότητα με το LangChain
FROM python:3.11-slim

# Ορίζουμε το working directory μέσα στο container
WORKDIR /app

# Αντιγράφουμε πρώτα το requirements.txt για να εκμεταλλευτούμε την cache του Docker
COPY requirements.txt .

# Εγκατάσταση των βιβλιοθηκών (χωρίς να κρατάμε άχρηστα αρχεία cache)
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγράφουμε όλο τον υπόλοιπο κώδικα (app.py, knowledge_base_airbnb.json κλπ)
COPY . .

# Ενημερώνουμε το Docker ότι το Streamlit χρησιμοποιεί την πόρτα 8501
EXPOSE 8501

# Εντολή εκκίνησης της εφαρμογής όταν τρέχει το container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
