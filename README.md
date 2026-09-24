# AI Document Assistant

A Streamlit chat application with face-recognition login and a FastAPI + RAG backend powered by Google Gemini and Supabase. Upload any document and ask questions — the AI answers **strictly from your uploaded document**, never from general knowledge.

---

## Features

- **Face ID** and **Email / Password** authentication backed by **Supabase** (with bcrypt password hashing)
- **Delete recent chats** option directly from the sidebar
- Upload **PDF, DOCX, PPTX, XLSX, CSV, TXT** documents
- Answers grounded exclusively in uploaded content via ChromaDB + Gemini
- Politely refuses off-topic and general-knowledge questions
- Chat history persists securely in Supabase cloud database across server restarts
- Each user's chats and documents are isolated from other users

---

## Project Structure

```
1st Project/
├── Frontend/
│   ├── app.py                  # Streamlit chat UI + face recognition login
│   └── requirements.txt
├── backend/
│   ├── main.py                 # FastAPI app (all endpoints)
│   ├── db.py                   # Supabase client initializer
│   ├── auth.py                 # Supabase user authentication & face embeddings
│   ├── parser.py               # Document text extraction
│   ├── rag.py                  # Chunking, embeddings, Chroma, Gemini answers
│   ├── storage.py              # Supabase chat & message persistence
│   ├── migrate_to_supabase.py  # One-time migration script (.pkl -> Supabase)
│   ├── requirements.txt
│   ├── .env.example            # Environment variables template
│   ├── storage/                # Uploaded files, one subfolder per chat_id
│   └── chroma_db/              # Vector store, one collection per chat_id
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── accounts_v3.pkl             # Local backup of user accounts (migrated to Supabase)
├── face_database_v3.pkl        # Local backup of face embeddings (migrated to Supabase)
└── README.md
```

---

## Setup & Supabase Configuration

### 1. Supabase Database Setup

1. Create a project at [https://supabase.com](https://supabase.com).
2. Go to the **SQL Editor** in the Supabase Dashboard and run the schema:

```sql
create table users (
    email text primary key,
    username text not null,
    password_hash text not null,
    created_at timestamptz default now()
);

create table face_embeddings (
    id bigserial primary key,
    email text references users(email) on delete cascade,
    embedding jsonb not null,
    created_at timestamptz default now()
);

create table chats (
    chat_id text primary key,
    user_email text references users(email) on delete cascade,
    title text not null,
    title_generated boolean not null default false,
    created_at timestamptz default now()
);

create table messages (
    id bigserial primary key,
    chat_id text references chats(chat_id) on delete cascade,
    role text not null,
    content text not null,
    created_at timestamptz default now()
);
```

### 2. Configure `backend/.env`

Create or edit `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_role_key_here
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
pip install -r Frontend/requirements.txt
```

### 4. Run One-Time Migration (from Local .pkl to Supabase)

To migrate existing accounts and face embeddings from `accounts_v3.pkl` and `face_database_v3.pkl` to Supabase:

```bash
python backend/migrate_to_supabase.py
```

Confirm that the output reports users and face embeddings migrated without errors. The original `.pkl` files remain untouched as backups.

---

## Running the App

Run both backend and frontend in two separate terminals:

### Terminal 1 — Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Check `http://localhost:8000/health` — you should see `{"status":"ok"}`.

### Terminal 2 — Frontend (Streamlit)

```bash
streamlit run Frontend/app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Test Flow

1. **Log in** with Face ID or Email & Password (migrated accounts log in seamlessly).
2. **Confirm chat history loads** — user chats are fetched directly from Supabase.
3. **Delete a recent chat** — click the `✕` delete button next to any chat in the sidebar to delete it from Supabase.
4. **Create a New Chat** — click "New Chat" in the sidebar.
5. **Upload a document** — upload a PDF/DOCX/TXT/PPTX/XLSX/CSV document in the chat input.
6. **Ask a grounded question** — verify the answer is grounded in the document.
7. **Ask an off-topic question** — e.g. *"What is the capital of France?"* — verify it is politely refused.
8. **Restart both servers** — confirm all chats and messages persist from Supabase.
9. **Test user isolation** — log in as a second user and verify each user sees only their own chats.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload & index a document (multipart: `chat_id`, `user_email`, `file`) |
| `POST` | `/chat` | Ask a question (JSON: `chat_id`, `user_email`, `question`) |
| `POST` | `/chats/create` | Create a new chat record in Supabase |
| `GET` | `/chats/{user_email}` | Get all chats for a user from Supabase |
| `DELETE` | `/chats/{chat_id}` | Delete a chat and its messages (cascades) |
| `POST` | `/chats/delete` | Delete a chat via JSON POST (`{"chat_id": "..."}`) |

Interactive docs: `http://localhost:8000/docs`
