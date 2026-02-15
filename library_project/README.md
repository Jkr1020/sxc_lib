# SXC Library Management

Flask-based library portal with:
- Student/staff login and signup
- Book search and recommendations
- Book request workflow (pending/approved/rejected)
- Admin dashboard with analytics and reports
- Student chatbot for library and college queries

## Run (Development)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Deploy (Production)

Use `wsgi.py` as entrypoint:
- Gunicorn: `gunicorn -c gunicorn.conf.py wsgi:app`
- Waitress (Windows/local production): `waitress-serve --host 0.0.0.0 --port 5000 wsgi:app`

See `DEPLOYMENT.md` for full setup.

