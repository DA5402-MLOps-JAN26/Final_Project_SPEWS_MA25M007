FROM python:3.10-slim

WORKDIR /app

RUN pip install streamlit plotly requests pandas --no-cache-dir

COPY . .

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:3000/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=3000", "--server.address=0.0.0.0"]
