# Deployment Guide

## 1) Production Readiness Checklist
- Set `SECRET_KEY` in environment variables.
- Set `FLASK_DEBUG=false`.
- Use managed DB in production by setting `DATABASE_URL` (PostgreSQL recommended).
- Keep `.env` out of source control.

## 2) Quick Local Production Run
From `library_project`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
waitress-serve --host 0.0.0.0 --port 5000 wsgi:app
```

Health check:

```text
GET /health
```

## 3) Render / Railway / Similar (Procfile-based)
- Build command:

```text
pip install -r requirements.txt
```

- Start command:

```text
gunicorn -c gunicorn.conf.py wsgi:app
```

- Required environment variables:
`SECRET_KEY`, optionally `DATABASE_URL`, and admin/staff vars from `.env.example`.

## 4) Docker

```bash
docker build -t sxc-library .
docker run --rm -p 8000:8000 --env-file .env sxc-library
```

## 5) Notes
- App bootstraps DB tables automatically on first request.
- If `DATABASE_URL` is not set, SQLite (`instance/library.db`) is used.
- For multi-instance hosting, use a shared external DB instead of SQLite.

