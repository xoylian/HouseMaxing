# Επίσημο Python 3.11 slim — σταθερό με LangChain + ChromaDB
FROM python:3.11-slim

# Απαραίτητα system packages για ChromaDB (χρειάζεται C compiler)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory μέσα στο container
WORKDIR /app

# Αντιγράφουμε πρώτα requirements για Docker cache layer optimization
COPY requirements.txt .

# Εγκατάσταση dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγράφουμε όλο τον κώδικα
COPY . .

# Streamlit config — απενεργοποιεί το browser auto-open και το telemetry
RUN mkdir -p /app/.streamlit && echo '\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /app/.streamlit/config.toml

# Port που χρησιμοποιεί το Streamlit
EXPOSE 8501

# Health check — ελέγχει ότι η εφαρμογή τρέχει
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Εκκίνηση
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
