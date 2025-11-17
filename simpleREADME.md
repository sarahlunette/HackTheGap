# TheLab – Crisis & Resilience AI

This project provides a crisis-planning AI using:
- Qdrant as the vectorstore
- FastAPI backend (`main_anthropic_qdrant.py`)
- Streamlit chat interface (`demo.py`)
- A build script to generate the vectorstore (`build_vectorstore.py`)

---

## 🚀 1. Start Qdrant

To launch Qdrant locally:

```bash
docker run -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Qdrant will now be available at:

```
http://localhost:6333
```

---

## 📦 2. Build the Vectorstore

Move into the `app/` directory:

```bash
cd app
```

Run the vectorstore builder:

```bash
python build_vectorstore.py
```

This reads documents from:

```
app/docs/
```

and creates a Qdrant collection.

---

## 🧠 3. Run the FastAPI Backend (Qdrant version)

From inside the `app/` directory:

```bash
uvicorn main_anthropic_qdrant:app --reload --port 9000
```

FastAPI will be available at:

```
http://localhost:9000
```

---

## 💬 4. Run the Streamlit Chat Interface

From the **project root**:

```bash
streamlit run streamlit/demo.py
```

This launches the chat UI that connects to the FastAPI backend.

---

## 📁 Directory Structure (relevant parts)

```
TheLab_/
├── app/
│   ├── main_anthropic_qdrant.py
│   ├── build_vectorstore.py
│   ├── docs/
│   └── vectorstore/
├── streamlit/
│   └── demo.py
└── requirements.txt
```

---

## ✔️ Summary

1. Start Qdrant  
2. Build vectorstore with `build_vectorstore.py`  
3. Run FastAPI using `main_anthropic_qdrant.py`  
4. Launch Streamlit chat UI  

Nothing else is required.

