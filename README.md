Backend (FastAPI):
  cd backend
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  uvicorn main:app --reload
Frontend (Vite + React):
  cd frontend
  npm install
  npm run dev
  open http://localhost:5173