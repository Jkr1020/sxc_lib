from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dataclasses import dataclass
import os
import pandas as pd
import io
import re
from fpdf import FPDF
import tempfile
import json
import difflib  # Added for fuzzy matching in chatbot
from datetime import datetime, timedelta, timezone  # For conversation timestamps
from html import escape as html_escape, unescape as html_unescape
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin, urlparse
from pathlib import Path
from sqlalchemy import inspect, text
import smtplib
import ssl
from email.message import EmailMessage
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

import requests
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = ""
except Exception as e:
    TfidfVectorizer = None
    linear_kernel = None
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = str(e)

try:
    from ml_engine import LibraryBrain as _RealLibraryBrain  # Import the new AI Brain
    LIBRARY_BRAIN_AVAILABLE = True
    LIBRARY_BRAIN_IMPORT_ERROR = ""
except Exception as e:
    _RealLibraryBrain = None
    LIBRARY_BRAIN_AVAILABLE = False
    LIBRARY_BRAIN_IMPORT_ERROR = str(e)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
BOOKS_CSV_PATH = BASE_DIR / "SXC_Library_Categorized new.csv"
ISSUES_CSV_PATH = BASE_DIR / "SXC library student issue  & return report.csv"

if not SKLEARN_AVAILABLE:
    print(f"[WARN] scikit-learn/scipy unavailable. Falling back to basic search. Details: {SKLEARN_IMPORT_ERROR}")

if not LIBRARY_BRAIN_AVAILABLE:
    print(f"[WARN] ml_engine unavailable. Falling back to basic library engine. Details: {LIBRARY_BRAIN_IMPORT_ERROR}")


class _FallbackLibraryBrain:
    def __init__(self, book_source: str, issue_source: Optional[str] = None):
        self.df = pd.DataFrame()
        try:
            self.df = pd.read_csv(book_source)
            if 'Title' not in self.df.columns:
                self.df = pd.DataFrame()
        except Exception:
            self.df = pd.DataFrame()
        if not self.df.empty:
            self.df['Title_Lower'] = self.df['Title'].astype(str).str.lower()

    def semantic_search(self, query: str, n: int = 10):
        q = (query or "").strip().lower()
        if not q or self.df.empty:
            return []
        matches = self.df[self.df['Title_Lower'].str.contains(q, na=False)].head(max(int(n or 10), 1))
        return [
            {
                'title': row.get('Title', ''),
                'category': row.get('Category', ''),
                'access_no': row.get('Access No', ''),
            }
            for _, row in matches.iterrows()
        ]

    def get_recommendations(self, current_title: str, n: int = 5):
        title = (current_title or "").strip().lower()
        if not title or self.df.empty:
            return []
        current = self.df[self.df['Title'].astype(str).str.lower() == title]
        if current.empty:
            return []
        category = str(current.iloc[0].get('Category', '') or '')
        recs = self.df[
            (self.df['Category'].astype(str) == category) &
            (self.df['Title'].astype(str).str.lower() != title)
        ].head(max(int(n or 5), 1))
        return [
            {
                'title': row.get('Title', ''),
                'category': row.get('Category', ''),
                'access_no': row.get('Access No', ''),
            }
            for _, row in recs.iterrows()
        ]

    def answer_question(self, *_args, **_kwargs):
        return "Library AI features are running in compatibility mode. Use search or request a book."


LibraryBrain = _RealLibraryBrain if LIBRARY_BRAIN_AVAILABLE else _FallbackLibraryBrain


@dataclass(frozen=True)
class CollegeSearchResult:
    url: str
    title: str
    score: float
    snippet: str


class _LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k and k.lower() == "href" and v:
                self.links.append(v)


def _utc_now():
    return datetime.now(timezone.utc)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _safe_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return html_unescape(title)


def _html_to_text(html: str) -> str:
    # Remove scripts/styles/comments
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    html = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)

    # Add line breaks for common block tags before stripping tags
    html = re.sub(r"</(p|div|br|li|h1|h2|h3|h4|tr|section|article)>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<br\\s*/?>", "\n", html, flags=re.IGNORECASE)

    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = html_unescape(text)

    # Normalize whitespace
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter that works decently on web text.
    parts = re.split(r"(?<=[.!?])\\s+|\\n\\n+", text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


class CollegeSiteKB:
    """
    Lightweight, self-contained "RAG" style knowledge base built by crawling the
    official SXC website and searching it with TF-IDF.
    """

    def __init__(
        self,
        base_url: str,
        cache_path: str,
        max_pages: int = 30,
        max_age_hours: int = 24,
        timeout_s: int = 10,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.cache_path = cache_path
        self.max_pages = max_pages
        self.max_age = timedelta(hours=max_age_hours)
        self.timeout_s = timeout_s

        parsed = urlparse(self.base_url)
        self._base_netloc = parsed.netloc.lower()
        self._user_agent = "SXC-LibBot/1.0 (+local; educational)"

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._docs: list[dict] = []

    def _is_internal(self, url: str) -> bool:
        p = urlparse(url)
        return p.netloc.lower() == self._base_netloc

    def _normalize_url(self, href: str, parent_url: str) -> Optional[str]:
        if not href:
            return None
        href = href.strip()
        if href.startswith("#"):
            return None
        if href.lower().startswith(("mailto:", "tel:", "javascript:")):
            return None

        abs_url = urljoin(parent_url, href)
        abs_url = abs_url.split("#", 1)[0]

        p = urlparse(abs_url)
        if not p.scheme.startswith("http"):
            return None
        if not self._is_internal(abs_url):
            return None

        # Skip obvious non-HTML assets
        if re.search(r"\\.(png|jpe?g|gif|webp|svg|css|js|ico|pdf|zip|rar)$", p.path, flags=re.IGNORECASE):
            return None

        return abs_url

    def _load_cache(self) -> Optional[dict]:
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _cache_is_fresh(self, cache: dict) -> bool:
        fetched_at = _parse_iso(cache.get("fetched_at", ""))
        if not fetched_at:
            return False
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)

        age = _utc_now() - fetched_at
        pages = cache.get("pages") or {}

        # If we previously cached *no* pages (e.g., temporary network issue),
        # retry sooner so college Q&A can recover automatically.
        if not pages:
            return age <= timedelta(minutes=15)

        return age <= self.max_age

    def refresh(self) -> None:
        visited: set[str] = set()
        queue: list[str] = [self.base_url]
        pages: dict[str, dict] = {}

        while queue and len(pages) < self.max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = requests.get(
                    url,
                    timeout=self.timeout_s,
                    headers={"User-Agent": self._user_agent},
                )
                if resp.status_code != 200:
                    continue

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    continue

                html = resp.text
            except Exception:
                continue

            title = _extract_title(html)
            text = _html_to_text(html)

            # Keep pages that have enough text to be useful
            if len(text) < 200:
                continue

            # Cap stored text to avoid ballooning the cache
            text = text[:50_000]

            pages[url] = {
                "url": url,
                "title": title,
                "text": text,
                "fetched_at": _safe_iso(_utc_now()),
            }

            # Enqueue internal links
            extractor = _LinkExtractor()
            try:
                extractor.feed(html)
            except Exception:
                extractor.links = []

            for href in extractor.links:
                next_url = self._normalize_url(href, url)
                if next_url and next_url not in visited and next_url not in queue:
                    queue.append(next_url)

        cache = {
            "base_url": self.base_url,
            "fetched_at": _safe_iso(_utc_now()),
            "pages": pages,
        }
        self._save_cache(cache)
        self._build_index(cache)

    def ensure_ready(self) -> None:
        cache = self._load_cache()
        if not cache or cache.get("base_url") != self.base_url or not self._cache_is_fresh(cache):
            self.refresh()
            return
        self._build_index(cache)

    def _build_index(self, cache: dict) -> None:
        pages = cache.get("pages") or {}
        docs = []
        for _, meta in pages.items():
            text = (meta or {}).get("text", "")
            if not text or len(text) < 200:
                continue
            docs.append(
                {
                    "url": (meta or {}).get("url", ""),
                    "title": (meta or {}).get("title", "") or (meta or {}).get("url", ""),
                    "text": text,
                }
            )

        self._docs = docs
        if not self._docs:
            self._vectorizer = None
            self._matrix = None
            return

        if not SKLEARN_AVAILABLE or TfidfVectorizer is None:
            self._vectorizer = None
            self._matrix = None
            return

        texts = [d["text"] for d in self._docs]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20_000)
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, n: int = 3) -> list[CollegeSearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        self.ensure_ready()
        if not self._docs:
            return []

        results: list[CollegeSearchResult] = []
        if not SKLEARN_AVAILABLE or not self._vectorizer or self._matrix is None or linear_kernel is None:
            terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
            scored = []
            for i, doc in enumerate(self._docs):
                text_lower = (doc.get("text") or "").lower()
                score = float(sum(text_lower.count(t) for t in terms))
                if score > 0:
                    scored.append((i, score))
            best = sorted(scored, key=lambda x: x[1], reverse=True)[:n]
            for idx, score in best:
                doc = self._docs[int(idx)]
                results.append(
                    CollegeSearchResult(
                        url=doc["url"],
                        title=doc["title"],
                        score=float(score),
                        snippet=self._best_snippet(query, doc["text"]),
                    )
                )
            return results

        q_vec = self._vectorizer.transform([query])
        scores = linear_kernel(q_vec, self._matrix).flatten()
        if scores.size == 0:
            return []

        best_idx = scores.argsort()[::-1][:n]
        for idx in best_idx:
            score = float(scores[idx])
            doc = self._docs[int(idx)]
            results.append(
                CollegeSearchResult(
                    url=doc["url"],
                    title=doc["title"],
                    score=score,
                    snippet=self._best_snippet(query, doc["text"]),
                )
            )
        return results

    def _best_snippet(self, query: str, text: str) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return (text or "")[:260]

        # Score sentences using the same vectorizer; keep it bounded.
        candidates = sentences[:200]
        try:
            sent_vec = self._vectorizer.transform(candidates) if self._vectorizer else None
            q_vec = self._vectorizer.transform([query]) if self._vectorizer else None
            if sent_vec is None or q_vec is None:
                raise ValueError("vectorizer not ready")
            sent_scores = linear_kernel(q_vec, sent_vec).flatten()
            best = int(sent_scores.argmax())
            snippet = candidates[best]
        except Exception:
            snippet = candidates[0]

        snippet = re.sub(r"\\s+", " ", snippet).strip()
        return snippet[:320]

    def answer(self, query: str) -> Optional[str]:
        results = self.search(query, n=2)
        if not results:
            return None

        lines = [
            "Here is what I found on the official St. Xavier's College website:",
            "",
        ]
        for r in results:
            title = html_escape(r.title or "Source")
            snippet = html_escape(r.snippet or "")
            url = html_escape(r.url)
            lines.append(
                f"<b>{title}</b><br>{snippet}<br><a href=\"{url}\" target=\"_blank\" rel=\"noopener\">Open source</a><br>"
            )

        # Render with HTML line breaks (the chat UI uses innerHTML)
        return "<br>".join(lines).strip()

app = Flask(__name__)
app.config["SECRET_KEY"] = (os.getenv("SECRET_KEY") or "change-this-in-production").strip()
app.secret_key = app.config["SECRET_KEY"]
os.makedirs(app.instance_path, exist_ok=True)

# --- DATABASE CONFIGURATION ---
db_path = Path(app.instance_path) / "library.db"
database_url = (os.getenv("DATABASE_URL") or "").strip()
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url or f"sqlite:///{db_path.as_posix()}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["JSON_SORT_KEYS"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = (os.getenv("SESSION_COOKIE_SAMESITE") or "Lax").strip()
app.config["SESSION_COOKIE_SECURE"] = _env_bool("SESSION_COOKIE_SECURE", default=False)
db = SQLAlchemy(app)

# --- USER MODEL ---
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'admin', 'staff', 'student'
    full_name = db.Column(db.String(150))
    email = db.Column(db.String(150))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        """Hash and set the password"""
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches the hashed password"""
        return check_password_hash(self.password, password)


class LoginEvent(db.Model):
    __tablename__ = 'login_events'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    attempted_role = db.Column(db.String(50))
    status = db.Column(db.String(20), nullable=False, default='success')
    note = db.Column(db.String(255))
    ip_address = db.Column(db.String(64))
    user_agent = db.Column(db.String(255))
    logged_in_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class BookRequest(db.Model):
    __tablename__ = 'book_requests'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(150))
    book_title = db.Column(db.String(255), nullable=False)
    access_no = db.Column(db.String(50))
    category = db.Column(db.String(150))
    requested_due_date = db.Column(db.String(10))
    status = db.Column(db.String(20), nullable=False, default='pending')  # pending, approved, rejected
    requested_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = db.Column(db.DateTime)
    reviewed_by = db.Column(db.String(100))
    note = db.Column(db.String(255))


def record_login_event(
    username: str,
    role: Optional[str],
    attempted_role: Optional[str],
    status: str,
    note: Optional[str] = None,
) -> None:
    try:
        forwarded_for = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        client_ip = forwarded_for or (request.remote_addr or "").strip() or None
        user_agent = (request.headers.get("User-Agent") or "").strip()[:255] or None
        ev = LoginEvent(
            username=(username or "").strip().lower() or "unknown",
            role=(role or "unknown").strip().lower(),
            attempted_role=(attempted_role or "").strip().lower() or None,
            status=(status or "failed").strip().lower(),
            note=(note or "").strip()[:255] or None,
            ip_address=client_ip,
            user_agent=user_agent,
        )
        db.session.add(ev)
        db.session.commit()
    except Exception as e:
        print(f"[WARN] LoginEvent save failed: {e}")
        db.session.rollback()

def append_login_to_transactions_csv(user: User) -> None:
    """Append a lightweight login transaction row so admin CSV-based views reflect current logins."""
    try:
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        username = (user.username or "").strip().lower() or "unknown"
        full_name = (user.full_name or "").strip() or username
        role = (user.role or "").strip().lower() or "unknown"

        required_cols = [
            "Member Code",
            "Member Name",
            "Access No",
            "Title",
            "Issue Date",
            "Due Date",
            "Return Date",
            "Document",
            "Staff Code",
        ]

        login_row = {
            "Member Code": username,
            "Member Name": full_name,
            "Access No": f"LOGIN-{now.strftime('%Y%m%d%H%M%S%f')}",
            "Title": "Portal Login",
            "Issue Date": date_str,
            "Due Date": date_str,
            "Return Date": date_str,
            "Document": "LOGIN",
            "Staff Code": role,
        }

        csv_path = Path(CSV_SOURCES.get("transactions", str(ISSUES_CSV_PATH)))
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=required_cols)
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
        df = pd.concat([df, pd.DataFrame([login_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # Keep in-memory dataframe aligned for endpoints that use global df_issues.
        global df_issues
        if isinstance(df_issues, pd.DataFrame):
            for col in required_cols:
                if col not in df_issues.columns:
                    df_issues[col] = ""
            df_issues = pd.concat([df_issues, pd.DataFrame([login_row])], ignore_index=True)
    except Exception as e:
        print(f"[WARN] append_login_to_transactions_csv failed: {e}")


# --- INITIALIZE DATABASE AT APP START ---
def init_database():
    """Create database tables and default users on app startup"""
    with app.app_context():
        db.create_all()
        try:
            inspector = inspect(db.engine)
            required_columns = {
                'users': {
                    'full_name': 'VARCHAR(150)',
                    'email': 'VARCHAR(150)',
                    'created_at': 'DATETIME',
                },
                'login_events': {
                    'attempted_role': 'VARCHAR(50)',
                    'status': "VARCHAR(20) DEFAULT 'success'",
                    'note': 'VARCHAR(255)',
                    'user_agent': 'VARCHAR(255)',
                },
                'book_requests': {
                    'role': 'VARCHAR(50)',
                    'email': 'VARCHAR(150)',
                    'access_no': 'VARCHAR(50)',
                    'category': 'VARCHAR(150)',
                    'requested_due_date': 'VARCHAR(10)',
                    'status': 'VARCHAR(20)',
                    'requested_at': 'DATETIME',
                    'reviewed_at': 'DATETIME',
                    'reviewed_by': 'VARCHAR(100)',
                    'note': 'VARCHAR(255)',
                },
            }

            with db.engine.begin() as conn:
                for table_name, col_defs in required_columns.items():
                    if not inspector.has_table(table_name):
                        continue
                    current_cols = {col.get('name') for col in inspector.get_columns(table_name)}
                    for col_name, col_type in col_defs.items():
                        if col_name in current_cols:
                            continue
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
                        print(f"[INFO] Added missing column {table_name}.{col_name}")
        except Exception as e:
            print(f"[WARN] Schema check/update failed: {e}")

        default_admin_user = (os.getenv("SXC_ADMIN_USERNAME") or "sxcadmin").strip().lower()
        default_staff_user = (os.getenv("SXC_STAFF_USERNAME") or "staff").strip().lower()

        default_password = (os.getenv("SXC_DEFAULT_PASSWORD") or "").strip().lower() or None
        default_admin_password = (os.getenv("SXC_ADMIN_PASSWORD") or default_password or "1923").strip().lower()
        default_staff_password = (os.getenv("SXC_STAFF_PASSWORD") or default_password or "1923").strip().lower()

        def upsert_user(username: str, role: str, full_name: str, password: str) -> bool:
            user = User.query.filter(db.func.lower(User.username) == username.lower()).first()
            changed = False
            if not user:
                user = User(username=username, role=role, full_name=full_name)
                user.set_password(password)
                db.session.add(user)
                return True

            normalized_username = (username or "").strip().lower()
            if normalized_username and user.username != normalized_username:
                conflict = (
                    User.query.filter(db.func.lower(User.username) == normalized_username)
                    .filter(User.id != user.id)
                    .first()
                )
                if not conflict:
                    user.username = normalized_username
                    changed = True

            if user.role != role:
                user.role = role
                changed = True

            if full_name and not (user.full_name or "").strip():
                user.full_name = full_name
                changed = True

            # Ensure default credentials always work
            if not user.check_password(password):
                user.set_password(password)
                changed = True

            return changed

        changed_any = False
        changed_any |= upsert_user(default_admin_user, "admin", "Administrator", default_admin_password)
        changed_any |= upsert_user(default_staff_user, "staff", "Staff Member", default_staff_password)

        # Optional sample student (auto-created students are supported at login)
        sample_student = (os.getenv("SXC_SAMPLE_STUDENT") or "23uma101").strip().lower()
        if re.match(r"^\d{2}[A-Za-z]{3}\d{3}$", sample_student):
            changed_any |= upsert_user(sample_student, "student", "Sample Student", sample_student)

        if changed_any:
            db.session.commit()
            print("[INFO] Database initialized/updated with default users")
        else:
            print("[INFO] Database already initialized")


_APP_BOOTSTRAPPED = False


def ensure_app_bootstrapped():
    global _APP_BOOTSTRAPPED
    if _APP_BOOTSTRAPPED:
        return
    init_database()
    _APP_BOOTSTRAPPED = True


# --- DATA LOADING ---
def load_data():
    try:
        df_b = pd.read_csv(BOOKS_CSV_PATH)
        df_b['Title_Lower'] = df_b['Title'].astype(str).str.lower()
    except:
        df_b = pd.DataFrame(columns=['Title', 'Category', 'Access No', 'Title_Lower'])

    try:
        df_i = pd.read_csv(ISSUES_CSV_PATH)
        df_i['Member Code Clean'] = df_i['Member Code'].astype(str).str.strip().str.lower()
    except:
        df_i = pd.DataFrame(columns=['Member Code', 'Member Code Clean', 'Member Name', 'Issue Date'])
    
    return df_b, df_i

df_books, df_issues = load_data()

def reload_data():
    global df_books, df_issues, brain
    df_books, df_issues = load_data()
    brain = LibraryBrain(str(BOOKS_CSV_PATH), str(ISSUES_CSV_PATH))
    print("[INFO] Data Reloaded Successfully")


# --- INITIALIZE BRAIN (UPGRADED) ---
# Now passing both files so the brain can calculate Popularity Scores
brain = LibraryBrain(str(BOOKS_CSV_PATH), str(ISSUES_CSV_PATH))
college_kb = CollegeSiteKB(
    base_url="https://stxavierstn.edu.in/",
    cache_path=os.path.join(app.instance_path, "college_site_cache.json"),
    max_pages=30,
    max_age_hours=24,
)

# --- CSV SOURCES FOR FILTERING ---
CSV_SOURCES = {
    'transactions': str(ISSUES_CSV_PATH),
    'books': str(BOOKS_CSV_PATH),
    'students': str(ISSUES_CSV_PATH)
}

# --- HELPER: EXTRACT DEPT FROM ROLL NO ---
def get_admin_insights():
    if df_issues.empty:
        return {}, {}, []

    def extract_dept(code):
        match = re.search(r'([a-zA-Z]+)', str(code))
        return match.group(1).upper() if match else 'Other'
    
    df_issues['Dept'] = df_issues['Member Code'].apply(extract_dept)
    dept_counts = df_issues['Dept'].value_counts().head(10).to_dict()

    try:    
        date_counts = df_issues['Issue Date'].value_counts().sort_index().tail(7).to_dict()
    except:
        date_counts = {}

    top_readers = df_issues['Member Name'].value_counts().head(5).reset_index().values.tolist()
    
    return dept_counts, date_counts, top_readers

# --- ROUTES ---

@app.before_request
def _bootstrap_once():
    ensure_app_bootstrapped()


@app.route('/')
def home():
    if 'user_id' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))


@app.route('/health', methods=['GET'])
def health():
    db_status = "ok"
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"

    return jsonify({
        "status": "ok" if db_status == "ok" else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database": db_status,
    })

@app.route('/init_db')
def init_db_route():
    """Initialize database and add default users"""
    init_database()
    return "Database initialized successfully!"

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = (request.args.get('success') or '').strip()
    selected_role = 'student'
    if request.method == 'POST':
        username_raw = (request.form.get('username') or '').strip()
        password_raw = (request.form.get('password') or '').strip()
        role_selection = (request.form.get('role') or 'student').strip().lower()
        if role_selection not in {'student', 'staff', 'admin'}:
            role_selection = 'student'
        selected_role = role_selection

        if not username_raw or not password_raw:
            return render_template(
                'login.html',
                error="Please enter username and password.",
                success=success,
                selected_role=selected_role,
            )

        default_admin_user = (os.getenv("SXC_ADMIN_USERNAME") or "sxcadmin").strip().lower()

        default_password = (os.getenv("SXC_DEFAULT_PASSWORD") or "").strip().lower() or None
        default_admin_password = (os.getenv("SXC_ADMIN_PASSWORD") or default_password or "1923").strip().lower()

        username = username_raw.strip().lower()
        password_input = password_raw.strip()

        def login_success(user: User):
            session['user_id'] = (user.username or "").strip().lower()
            session['role'] = user.role
            session['full_name'] = user.full_name
            append_login_to_transactions_csv(user)
            record_login_event(
                username=(user.username or "").strip().lower(),
                role=user.role,
                attempted_role=role_selection,
                status="success",
                note="Login successful",
            )

            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('student_dashboard'))

        user = User.query.filter(db.func.lower(User.username) == username).first()
        if role_selection == "admin" and username == default_admin_user:
            if not user:
                user = User(username=default_admin_user, role="admin", full_name="Administrator")
                user.set_password(default_admin_password)
                db.session.add(user)
                db.session.commit()

            if user.check_password(password_input):
                return login_success(user)
            error = "Invalid admin login. Check your credentials."
        else:
            if user and user.check_password(password_input):
                if user.role != role_selection:
                    error = f"Role mismatch. This account is registered as {user.role}."
                else:
                    return login_success(user)

            if user:
                error = error or "Invalid login. Check your credentials."
            else:
                if role_selection == "admin":
                    error = "Admin account not found. Check admin credentials."
                else:
                    error = f"{role_selection.title()} account not found. Please sign up first."

        if error:
            resolved_role = user.role if ('user' in locals() and user) else role_selection
            record_login_event(
                username=username,
                role=resolved_role,
                attempted_role=role_selection,
                status="failed",
                note=error,
            )

    default_admin_user = (os.getenv("SXC_ADMIN_USERNAME") or "sxcadmin").strip().lower()

    default_password = (os.getenv("SXC_DEFAULT_PASSWORD") or "").strip().lower() or None
    default_admin_password = (os.getenv("SXC_ADMIN_PASSWORD") or default_password or "1923").strip().lower()

    return render_template(
        'login.html',
        error=error,
        success=success,
        selected_role=selected_role,
        admin_username=default_admin_user,
        admin_password=default_admin_password,
    )


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    error = None
    success = (request.args.get('success') or '').strip()

    if request.method == 'POST':
        identifier = (request.form.get('identifier') or '').strip().lower()
        if not identifier:
            error = "Enter your username or registered email."
        else:
            user = (
                User.query.filter(db.func.lower(User.username) == identifier).first()
                or User.query.filter(db.func.lower(db.func.coalesce(User.email, '')) == identifier).first()
            )

            if user and (user.email or '').strip():
                try:
                    token = _create_password_reset_token(user)
                    subject, body = _build_password_reset_email(user, token)
                    _send_email((user.email or '').strip(), subject, body)
                except Exception as e:
                    print(f"[WARN] Forgot-password email flow failed: {e}")

            success = "If the account exists and has a registered email, a reset link has been sent."

    return render_template('forgot_password.html', error=error, success=success)


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token: str):
    error = None
    success = None
    user = _resolve_password_reset_token(token)

    if not user:
        return render_template(
            'reset_password.html',
            error="This reset link is invalid or expired. Request a new password reset link.",
            success=success,
            token_valid=False,
            token=token,
        )

    if request.method == 'POST':
        password_raw = (request.form.get('password') or '').strip()
        confirm_password = (request.form.get('confirm_password') or '').strip()

        if not password_raw or not confirm_password:
            error = "Enter and confirm your new password."
        elif len(password_raw) < 6:
            error = "Password must be at least 6 characters."
        elif password_raw != confirm_password:
            error = "Passwords do not match."
        else:
            try:
                user.set_password(password_raw)
                db.session.commit()
                return redirect(url_for('login', success='Password reset successful. You can login now.'))
            except Exception as e:
                db.session.rollback()
                error = f"Failed to reset password: {e}"

    return render_template(
        'reset_password.html',
        error=error,
        success=success,
        token_valid=True,
        token=token,
    )


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    success = None
    selected_role = 'student'

    if request.method == 'POST':
        full_name = (request.form.get('full_name') or '').strip()
        username = (request.form.get('username') or request.form.get('roll_no') or '').strip().lower()
        dob = (request.form.get('dob') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password_raw = (request.form.get('password') or '').strip()
        role = (request.form.get('role') or 'student').strip().lower()
        if role not in {'student', 'staff'}:
            role = 'student'
        selected_role = role

        if not full_name or not username or not dob or not email or not password_raw:
            error = "All fields are required."
        elif not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
            error = "Enter a valid email address."
        else:
            try:
                datetime.strptime(dob, "%Y-%m-%d")
            except Exception:
                error = "DOB must be in YYYY-MM-DD format."

        if not error:
            try:
                existing_user = User.query.filter(db.func.lower(User.username) == username).first()
                if existing_user:
                    error = "This username is already registered. Please login."

                if not error:
                    user = User(
                        username=username,
                        role=role,
                        full_name=full_name,
                        email=email,
                    )
                    user.set_password(password_raw)
                    db.session.add(user)
                    db.session.commit()
                    return redirect(url_for('login', success='Signup successful. You can login now.'))
            except Exception as e:
                db.session.rollback()
                error = f"Signup failed: {e}"

    return render_template('signup.html', error=error, success=success, selected_role=selected_role)

# --- ADD THIS ROUTE TO app.py ---

@app.route('/api/random', methods=['GET'])
def get_random_books():
    if df_books.empty: return jsonify([])
    
    # Pick 6 random books safely
    count = min(6, len(df_books))
    sample = df_books.sample(n=count).fillna('')
    
    results = []
    for _, row in sample.iterrows():
        results.append({
            'title': row['Title'],
            'category': row['Category']
        })
    return jsonify(results)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- SEARCH ENGINE & DASHBOARD (UPGRADED) ---

@app.route('/dashboard')
def student_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    return render_template('index.html', user_id=session['user_id'], role=session.get('role'))

@app.route('/api/search', methods=['POST'])
def search_books():
    query = request.json.get('query', '').lower()
    if not query: return jsonify([])

    # 1. Try Semantic Search (AI)
    ai_results = brain.semantic_search(query)
    if ai_results:
        return jsonify(ai_results)

    # 2. Fallback to Simple Keyword Matching
    if not df_books.empty:
        matches = df_books[df_books['Title_Lower'].str.contains(query, na=False)].head(10)
        results = []
        for _, row in matches.iterrows():
            results.append({'title': row['Title'], 'category': row['Category'], 'access_no': row['Access No']})
        return jsonify(results)
    
    return jsonify([])

@app.route('/api/recommend', methods=['POST'])
def recommend_books():
    payload = request.get_json(silent=True) or {}
    current_title = payload.get('current_title') or payload.get('currentTitle') or payload.get('title')
    category = payload.get('category')

    # Track last selected book for better chatbot follow-ups
    if current_title:
        session['last_book'] = current_title

    # 1) Content-based recommendations (AI)
    if current_title:
        ai_recs = brain.get_recommendations(current_title)
        if ai_recs:
            return jsonify(ai_recs)

    # 2) Fallback: same category
    if not category or df_books.empty:
        return jsonify([])

    recommendations = df_books[(df_books['Category'] == category) & (df_books['Title'] != current_title)]
    if recommendations.empty:
        return jsonify([])

    sample = recommendations.sample(n=min(5, len(recommendations)), replace=False).fillna('')
    results = []
    for _, row in sample.iterrows():
        results.append({
            'title': row.get('Title', ''),
            'category': row.get('Category', ''),
            'access_no': row.get('Access No', '')
        })
    return jsonify(results)


@app.route('/api/book_requests', methods=['POST'])
def create_book_request():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    title = (payload.get('title') or '').strip()
    access_no = (payload.get('access_no') or payload.get('accessNo') or '').strip()
    category = (payload.get('category') or '').strip()
    email = (payload.get('email') or '').strip()

    if not title:
        return jsonify({'success': False, 'error': 'Book title is required.'}), 400
    if not email:
        return jsonify({'success': False, 'error': 'Email is required to request a book.'}), 400

    # Request due date is auto-assigned to 10 days from request date.
    requested_due_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")

    username = (session.get('user_id') or '').strip().lower()
    role = (session.get('role') or 'student').strip().lower()

    try:
        # Update user email if provided
        if email:
            user = User.query.filter(db.func.lower(User.username) == username).first()
            if user and (user.email or "").strip().lower() != email.lower():
                user.email = email
                db.session.commit()

        existing = (
            BookRequest.query.filter(db.func.lower(BookRequest.username) == username)
            .filter(BookRequest.status == 'pending')
            .filter(BookRequest.book_title == title)
            .filter(BookRequest.access_no == access_no)
            .first()
        )
        if existing:
            return jsonify({'success': False, 'error': 'This request is already pending.'}), 400

        req = BookRequest(
            username=username,
            role=role,
            email=email or None,
            book_title=title,
            access_no=access_no or None,
            category=category or None,
            requested_due_date=requested_due_date,
            status='pending',
        )
        db.session.add(req)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/book_requests/my', methods=['GET'])
def get_my_book_requests():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    username = (session.get('user_id') or '').strip().lower()
    try:
        rows = (
            BookRequest.query.filter(db.func.lower(BookRequest.username) == username)
            .order_by(BookRequest.requested_at.desc())
            .limit(50)
            .all()
        )
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "book_title": r.book_title,
                "access_no": r.access_no or "",
                "category": r.category or "",
                "requested_due_date": r.requested_due_date or "",
                "status": r.status,
                "requested_at": r.requested_at.strftime("%d-%m-%Y %H:%M") if r.requested_at else "",
                "note": r.note or "",
            })
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _format_chat_text(text: str) -> str:
    """Escape as HTML and apply small formatting (bold + newlines) for chat UI (innerHTML)."""
    safe = html_escape(text or "")
    # Minimal markdown-to-HTML: **bold**
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return safe.replace("\n", "<br>")


def _get_smtp_config() -> dict:
    raw_port = (os.getenv("SMTP_PORT") or "587").strip()
    try:
        port = int(raw_port)
    except Exception:
        port = 587

    return {
        "host": (os.getenv("SMTP_HOST") or "").strip(),
        "port": port,
        "user": (os.getenv("SMTP_USER") or "").strip(),
        "password": (os.getenv("SMTP_PASSWORD") or "").strip(),
        "from_addr": (os.getenv("SMTP_FROM") or "").strip(),
        "use_tls": (os.getenv("SMTP_TLS") or "true").strip().lower() in ["1", "true", "yes", "y"],
    }


def _send_email(to_addr: str, subject: str, body: str) -> tuple[bool, str, str]:
    to_addr = (to_addr or "").strip()
    if not to_addr:
        print("[WARN] Recipient email missing; email skipped.")
        return False, "email_missing", "Recipient email is missing."

    cfg = _get_smtp_config()
    missing_cfg = [k for k in ["host", "user", "password", "from_addr"] if not cfg.get(k)]
    if missing_cfg:
        print(f"[WARN] SMTP not configured; missing: {', '.join(missing_cfg)}")
        return False, "smtp_not_configured", f"Missing SMTP fields: {', '.join(missing_cfg)}"

    msg = EmailMessage()
    msg["From"] = cfg["from_addr"]
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as smtp:
            smtp.ehlo()
            if cfg["use_tls"]:
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            smtp.login(cfg["user"], cfg["password"])
            smtp.send_message(msg)
        return True, "sent", ""
    except smtplib.SMTPAuthenticationError as e:
        raw = ""
        try:
            raw = (e.smtp_error or b"").decode("utf-8", errors="ignore")
        except Exception:
            raw = str(e)
        detail = (raw or str(e) or "SMTP authentication failed.").strip()
        print(f"[WARN] Email send failed (auth): {detail}")
        return False, "send_failed", f"SMTP auth failed: {detail[:220]}"
    except Exception as e:
        print(f"[WARN] Email send failed: {e}")
        detail = (str(e) or "Unknown SMTP error").strip()
        return False, "send_failed", detail[:220]


def _get_password_reset_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="sxc-password-reset")


def _get_password_reset_ttl_seconds() -> int:
    raw = (os.getenv("PASSWORD_RESET_TTL_SECONDS") or "1800").strip()
    try:
        ttl = int(raw)
        if ttl < 300:
            return 300
        if ttl > 86400:
            return 86400
        return ttl
    except Exception:
        return 1800


def _create_password_reset_token(user: User) -> str:
    serializer = _get_password_reset_serializer()
    payload = {
        "uid": int(user.id),
        "pw": (user.password or ""),
    }
    return serializer.dumps(payload)


def _resolve_password_reset_token(token: str) -> Optional[User]:
    if not token:
        return None

    serializer = _get_password_reset_serializer()
    try:
        payload = serializer.loads(token, max_age=_get_password_reset_ttl_seconds())
    except (BadSignature, SignatureExpired):
        return None

    uid = payload.get("uid")
    password_marker = payload.get("pw")
    if uid is None:
        return None

    user = User.query.get(uid)
    if not user:
        return None
    if (user.password or "") != (password_marker or ""):
        # Token becomes invalid after password change.
        return None
    return user


def _build_password_reset_email(user: User, token: str) -> tuple[str, str]:
    name = (user.full_name or user.username or "User").strip()
    reset_link = url_for("reset_password", token=token, _external=True)
    ttl_minutes = max(1, _get_password_reset_ttl_seconds() // 60)

    subject = "SXC Library | Password Reset"
    body = (
        f"Hello {name},\n\n"
        "We received a request to reset your SXC Library account password.\n\n"
        f"Reset password link:\n{reset_link}\n\n"
        f"This link will expire in {ttl_minutes} minutes.\n"
        "If you did not request this, please ignore this email.\n\n"
        "Regards,\nSXC Library Admin\n"
    )
    return subject, body


SXC_COLLEGE_NAME = "St. Xavier's College (Autonomous), Palayamkottai"
SXC_COLLEGE_SINCE = "1923"
SXC_COLLEGE_WEBSITE = "https://stxavierstn.edu.in/"
SXC_DEPARTMENTS = [
    ("UCS", "Computer Science"),
    ("UDS", "Data Science"),
    ("UFS", "Forensic Science"),
    ("UMA", "Mathematics"),
    ("UPH", "Physics"),
    ("UCH", "Chemistry"),
    ("UBO", "Botany"),
    ("UZO", "Zoology"),
    ("UEN", "English"),
    ("UTA", "Tamil"),
    ("COM", "BBA/Commerce"),
    ("GEO", "Geography"),
]


def _college_fallback_html(query: str) -> Optional[str]:
    q = (query or "").strip().lower()
    if not q:
        return None

    site = html_escape(SXC_COLLEGE_WEBSITE)

    if any(k in q for k in ["department", "departments", "courses", "course", "program", "programmes"]):
        dept_lines = "<br>".join([f"<b>{html_escape(code)}</b> — {html_escape(name)}" for code, name in SXC_DEPARTMENTS])
        return (
            f"<b>Departments (from SXC Library system)</b><br><br>"
            f"{dept_lines}<br><br>"
            f"For the official and latest details, please visit: "
            f"<a href=\"{site}\" target=\"_blank\" rel=\"noopener\">{site}</a>"
        )

    if any(k in q for k in ["history", "about", "established", "founded", "since", "autonomous", "palayamkottai"]):
        return (
            f"<b>{html_escape(SXC_COLLEGE_NAME)}</b><br><br>"
            f"Established: <b>{html_escape(SXC_COLLEGE_SINCE)}</b><br>"
            f"Location: Palayamkottai, Tamil Nadu, India<br><br>"
            f"Official website: <a href=\"{site}\" target=\"_blank\" rel=\"noopener\">{site}</a>"
        )

    if any(k in q for k in ["admission", "apply", "application", "eligibility", "fees", "fee", "prospectus"]):
        return (
            "<b>Admissions / Fees</b><br><br>"
            "For up-to-date admission notifications, eligibility and fee details, please refer to the official website:<br>"
            f"<a href=\"{site}\" target=\"_blank\" rel=\"noopener\">{site}</a>"
        )

    if any(k in q for k in ["contact", "address", "phone", "email", "office"]):
        return (
            "<b>College Contact</b><br><br>"
            "Please check the official website contact page for the latest phone/email/address details:<br>"
            f"<a href=\"{site}\" target=\"_blank\" rel=\"noopener\">{site}</a>"
        )

    return None


def _answer_college_best_effort(query: str) -> Optional[str]:
    try:
        answer = college_kb.answer(query)
        if answer:
            return answer
    except Exception as e:
        print(f"[WARN] College KB Error: {e}")

    return _college_fallback_html(query)


# --- CHATBOT API ---
@app.route('/api/chat', methods=['POST'])
def chatbot():
    payload = request.get_json(silent=True) or {}
    msg_raw = (payload.get('message') or '').strip()
    role = session.get('role') or 'guest'
    user_id = session.get('user_id', 'Guest')

    if not msg_raw:
        return jsonify({'response': _format_chat_text('Please ask me something! 😊')})

    session.setdefault('conversation_history', [])
    session.setdefault('last_book', None)

    msg_lower = msg_raw.lower()

    # 1) Student borrowing status
    if any(word in msg_lower for word in ['my books', 'borrowed', 'borrow status', 'due date', 'return date', 'issued', 'transactions']):
        if role != 'student':
            response = "🔐 Please login as a student to view your borrowing status."
        else:
            user_issues = df_issues[df_issues['Member Code Clean'] == str(user_id).lower()]
            if not user_issues.empty:
                recent = user_issues.tail(5)
                lines = ["📚 Your Borrowing History (latest 5):", ""]
                for idx, (_, row) in enumerate(recent.iterrows(), 1):
                    lines.append(f"{idx}. {row.get('Title', '')}")
                    lines.append(f"   Issue: {row.get('Issue Date', '')} | Due: {row.get('Due Date', '')}")
                    lines.append("")
                lines.append(f"Total Records: {len(user_issues)}")
                response = "\n".join(lines).strip()
            else:
                response = "✅ You have no borrowing records yet!"

        session['conversation_history'].append({'user': msg_raw, 'bot': response})
        return jsonify({'response': _format_chat_text(response)})

    # 2) Fast FAQs (kept explicit so answers are always consistent)
    if any(word in msg_lower for word in ['timing', 'hours', 'open', 'close']):
        response = "📅 Library hours: 8:00 AM to 6:00 PM (Monday to Saturday)."
        session['conversation_history'].append({'user': msg_raw, 'bot': response})
        return jsonify({'response': _format_chat_text(response)})

    if any(word in msg_lower for word in ['fine', 'penalty', 'late fee']):
        response = "💰 Late return fine: Rs. 1 per day after the due date."
        session['conversation_history'].append({'user': msg_raw, 'bot': response})
        return jsonify({'response': _format_chat_text(response)})

    if any(word in msg_lower for word in ['borrow limit', 'borrowing rules', 'rules']):
        response = "📖 Borrowing rule: Students can borrow up to 2 books for 15 days."
        session['conversation_history'].append({'user': msg_raw, 'bot': response})
        return jsonify({'response': _format_chat_text(response)})

    # 2.5) Availability (best-effort from transactions; inventory doesn't store copy counts)
    if any(word in msg_lower for word in ['available', 'availability', 'copies', 'in stock']):
        try:
            hits = brain.semantic_search(msg_raw, n=1)
            if hits and hits[0].get('relevance', 0) >= 25:
                title = hits[0].get('title', '')
                active_issues = 0
                try:
                    if not df_issues.empty and 'Title' in df_issues.columns:
                        title_match = df_issues['Title'].fillna("").astype(str).str.strip().str.lower().eq(str(title).strip().lower())
                        if 'Return Date' in df_issues.columns:
                            ret = df_issues['Return Date'].fillna("").astype(str).str.strip()
                            is_active = ret.eq("")
                        else:
                            is_active = pd.Series([False] * len(df_issues))
                        active_issues = int((title_match & is_active).sum())
                except Exception:
                    active_issues = 0

                response = (
                    f"**Availability (from transactions):**\n\n"
                    f"**{title}**\n"
                    f"Active issues: **{active_issues}**\n\n"
                    "Note: The book inventory CSV doesn't include copy counts, so exact available copies can't be confirmed."
                )
                session['conversation_history'].append({'user': msg_raw, 'bot': response})
                return jsonify({'response': _format_chat_text(response)})
        except Exception as e:
            print(f"[WARN] Availability check error: {e}")

    # 3) Detect college vs library
    library_transaction_keywords = [
        'issue', 'issued', 'return', 'returned', 'due', 'overdue', 'borrow', 'renew', 'fine',
        'penalty', 'late fee', 'access no', 'member', 'my books', 'borrowed', 'transactions',
    ]
    library_discovery_keywords = [
        'library', 'book', 'books', 'catalog', 'catalogue', 'recommend', 'suggest', 'search',
        'find', 'availability', 'copies', 'popular', 'top books', 'category', 'categories',
        'statistics', 'stats',
    ]
    college_keywords = [
        'st xavier', 'st. xavier', 'stxavier', 'sxc', 'palayamkottai',
        'college', 'history', 'since', 'established', 'founded', 'vision', 'mission',
        'admission', 'course', 'courses', 'program', 'programmes',
        'department', 'departments', 'principal', 'contact', 'address', 'phone', 'email', 'office',
        'fee', 'fees', 'syllabus', 'autonomous', 'naac', 'nirf', 'placement', 'hostel',
        'iqac', 'research', 'alumni', 'campus', 'library', 'librarian',
    ]
    college_strong_keywords = [
        'history', 'admission', 'principal', 'naac', 'nirf', 'placement', 'hostel', 'contact',
        'address', 'phone', 'email', 'office', 'fee', 'fees', 'syllabus', 'vision', 'mission',
        'palayamkottai', 'iqac',
    ]

    looks_like_library_txn = any(k in msg_lower for k in library_transaction_keywords)
    looks_like_library = looks_like_library_txn or any(k in msg_lower for k in library_discovery_keywords)
    looks_like_college = any(k in msg_lower for k in college_keywords)
    looks_like_college_strong = any(k in msg_lower for k in college_strong_keywords)

    dept_query = any(k in msg_lower for k in ['department', 'departments'])
    dept_library_context = any(k in msg_lower for k in ['library', 'borrow', 'issue', 'transactions', 'usage', 'stats', 'statistics'])
    if dept_query and not dept_library_context:
        looks_like_college = True
        looks_like_college_strong = True

    # 4) If it looks like college info, try the official website KB first
    if (looks_like_college_strong and not looks_like_library_txn) or (looks_like_college and not looks_like_library):
        answer = _answer_college_best_effort(msg_raw)
        if answer:
            session['conversation_history'].append({'user': msg_raw, 'bot': answer})
            return jsonify({'response': answer})

    # 5) Library AI engine
    try:
        answer = brain.answer_question(
            msg_raw,
            user_id=user_id,
            user_role=role,
            last_book=session.get('last_book'),
        )
        if answer:
            def _is_generic_library_fallback(text: str) -> bool:
                t = (text or "").lower()
                if "i am designed to assist only" in t:
                    return True
                if "welcome to libbot" in t and "book search" in t and "library statistics" in t:
                    return True
                if "what would you like to know about our library" in t:
                    return True
                return False

            # If this looks like a college/general query and the library engine returned a generic fallback,
            # try the official website KB for a better answer.
            if looks_like_college and not looks_like_library_txn and _is_generic_library_fallback(answer):
                college_answer = _answer_college_best_effort(msg_raw)
                if college_answer:
                    session['conversation_history'].append({'user': msg_raw, 'bot': college_answer})
                    return jsonify({'response': college_answer})

            # If this was a search, track the last_book for better follow-ups.
            if any(w in msg_lower for w in ['search', 'find', 'look for']):
                results = brain.semantic_search(msg_raw, n=1)
                if results and results[0].get('relevance', 0) >= 25:
                    session['last_book'] = results[0].get('title')

            session['conversation_history'].append({'user': msg_raw, 'bot': answer})
            return jsonify({'response': _format_chat_text(answer)})
    except Exception as e:
        print(f"[WARN] Chatbot Error: {e}")

    # 6) Fallback: try website KB if this looks like college info
    if looks_like_college and not looks_like_library_txn:
        answer = _answer_college_best_effort(msg_raw)
        if answer:
            session['conversation_history'].append({'user': msg_raw, 'bot': answer})
            return jsonify({'response': answer})

    response = (
        "LibBot — St. Xavier's College (Autonomous), Palayamkottai\n\n"
        "I can help with:\n"
        "• Library: book search, recommendations, issues/returns, due dates, fines, stats\n"
        "• College: official info from the website (history, departments, courses, admissions, contact)\n\n"
        "Try asking:\n"
        "• find tamil books\n"
        "• my books / due date\n"
        "• college history\n"
        "• departments / courses\n"
        "• college address / contact"
    )
    session['conversation_history'].append({'user': msg_raw, 'bot': response})
    return jsonify({'response': _format_chat_text(response)})


# --- ADMIN PORTAL ---
@app.route('/admin')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    dept_counts, date_counts, top_readers = get_admin_insights()
    categories = sorted(df_books['Category'].dropna().astype(str).unique().tolist()) if not df_books.empty else []

    # Compute overdue count (active issues past due date)
    overdue_count = 0
    active_issues = 0
    returned_issues = 0
    return_rate = 0.0
    latest_issue_date = ""
    try:
        if not df_issues.empty:
            issue_dt = pd.to_datetime(df_issues.get('Issue Date'), dayfirst=True, errors='coerce')
            due = pd.to_datetime(df_issues.get('Due Date'), dayfirst=True, errors='coerce')
            ret = pd.to_datetime(df_issues.get('Return Date'), dayfirst=True, errors='coerce')

            active_issues = int(ret.isna().sum())
            returned_issues = int(ret.notna().sum())
            total = int(len(df_issues))
            return_rate = round((returned_issues / total) * 100, 1) if total else 0.0

            if issue_dt.notna().any():
                latest_issue_date = issue_dt.max().strftime('%d-%m-%Y')

        if not df_issues.empty and 'Due Date' in df_issues.columns:
            due = pd.to_datetime(df_issues['Due Date'], dayfirst=True, errors='coerce')
            now = pd.Timestamp.now().normalize()
            overdue_count = int(((ret.isna()) & (due.notna()) & (due < now)).sum())
    except Exception:
        overdue_count = 0
        active_issues = 0
        returned_issues = 0
        return_rate = 0.0
        latest_issue_date = ""

    # Top borrowed books (for quick insights card)
    top_books = []
    try:
        if not df_issues.empty and 'Title' in df_issues.columns:
            vc = df_issues['Title'].astype(str).value_counts().head(5)
            top_books = [[t, int(c)] for t, c in vc.items()]
    except Exception:
        top_books = []
    
    stats = {
        'total_issues': len(df_issues),
        'unique_students': df_issues['Member Code'].nunique() if not df_issues.empty else 0,
        'unique_books': df_books['Title'].nunique() if not df_books.empty else 0,
        'overdue_count': overdue_count,
        'active_issues': active_issues,
        'returned_issues': returned_issues,
        'return_rate': return_rate,
        'latest_issue_date': latest_issue_date,
    }
    
    # Get latest transactions data (newest first) with status + fine.
    transactions = []
    if not df_issues.empty:
        try:
            transactions_data = df_issues.copy()
            for col in ["Issue Date", "Due Date", "Return Date"]:
                if col not in transactions_data.columns:
                    transactions_data[col] = pd.NaT
            transactions_data["_issue_dt"] = pd.to_datetime(transactions_data["Issue Date"], dayfirst=True, errors="coerce")
            transactions_data["_due_dt"] = pd.to_datetime(transactions_data["Due Date"], dayfirst=True, errors="coerce")
            transactions_data["_ret_dt"] = pd.to_datetime(transactions_data["Return Date"], dayfirst=True, errors="coerce")

            now = pd.Timestamp.now().normalize()
            transactions_data["Status"] = "Active"
            transactions_data.loc[transactions_data["_ret_dt"].notna(), "Status"] = "Returned"
            transactions_data.loc[
                transactions_data["_ret_dt"].isna()
                & transactions_data["_due_dt"].notna()
                & (transactions_data["_due_dt"] < now),
                "Status",
            ] = "Overdue"

            ref_date = transactions_data["_ret_dt"].fillna(now)
            late_days = (ref_date.dt.normalize() - transactions_data["_due_dt"].dt.normalize()).dt.days
            transactions_data["Fine"] = late_days.where(late_days > 0, 0).fillna(0).astype(int)

            transactions_data = transactions_data.sort_values(["_issue_dt"], ascending=[False], na_position="last").head(10)
            transactions_data["Issue Date"] = transactions_data["_issue_dt"].dt.strftime("%d-%m-%Y").fillna("")
            transactions_data["Due Date"] = transactions_data["_due_dt"].dt.strftime("%d-%m-%Y").fillna("")
            transactions_data["Return Date"] = transactions_data["_ret_dt"].dt.strftime("%d-%m-%Y").fillna("")
            transactions = transactions_data.fillna("").to_dict("records")
        except Exception:
            transactions = []

    recent_logins = []
    try:
        rows = LoginEvent.query.order_by(LoginEvent.logged_in_at.desc()).limit(10).all()
        for ev in rows:
            recent_logins.append(
                {
                    "logged_in_at": ev.logged_in_at.strftime("%d-%m-%Y %H:%M") if ev.logged_in_at else "",
                    "username": (ev.username or "").strip().lower(),
                    "role": (ev.role or "").strip().lower(),
                    "status": (ev.status or "success").strip().lower(),
                    "ip_address": (ev.ip_address or "").strip(),
                }
            )
    except Exception:
        recent_logins = []

    return render_template('admin.html', stats=stats, 
                           dept_data=json.dumps(dept_counts), 
                           date_data=json.dumps(date_counts),
                           top_readers=top_readers,
                           top_books=top_books,
                           recent_logins=recent_logins,
                           categories=categories,
                           transactions=transactions,
                           user_id=session.get('full_name') or session.get('user_id') or 'Admin',
                           role=session.get('role') or 'admin')

# --- ADMIN API ENDPOINTS ---
@app.route('/api/admin/transactions', methods=['GET'])
def get_transactions():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        if df_issues.empty:
            return jsonify({'success': True, 'data': [], 'total': 0})

        result = df_issues.copy()
        for col in ['Issue Date', 'Due Date', 'Return Date']:
            if col not in result.columns:
                result[col] = pd.NaT
        result['_issue_dt'] = pd.to_datetime(result['Issue Date'], dayfirst=True, errors='coerce')
        result['_due_dt'] = pd.to_datetime(result['Due Date'], dayfirst=True, errors='coerce')
        result['_ret_dt'] = pd.to_datetime(result['Return Date'], dayfirst=True, errors='coerce')

        now = pd.Timestamp.now().normalize()
        result['Status'] = 'Active'
        result.loc[result['_ret_dt'].notna(), 'Status'] = 'Returned'
        result.loc[result['_ret_dt'].isna() & result['_due_dt'].notna() & (result['_due_dt'] < now), 'Status'] = 'Overdue'

        ref_date = result['_ret_dt'].fillna(now)
        late_days = (ref_date.dt.normalize() - result['_due_dt'].dt.normalize()).dt.days
        result['Fine'] = late_days.where(late_days > 0, 0).fillna(0).astype(int)

        result = result.sort_values(['_issue_dt'], ascending=[False], na_position='last').head(100)
        result['Issue Date'] = result['_issue_dt'].dt.strftime('%d-%m-%Y').fillna("")
        result['Due Date'] = result['_due_dt'].dt.strftime('%d-%m-%Y').fillna("")
        result['Return Date'] = result['_ret_dt'].dt.strftime('%d-%m-%Y').fillna("")

        transactions = result.fillna("").to_dict(orient='records')
        return jsonify({'success': True, 'data': transactions, 'total': len(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/books', methods=['GET'])
def get_books():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get all books from df_books
        books = df_books.head(100).to_dict(orient='records')
        return jsonify({'success': True, 'data': books, 'total': len(df_books)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students', methods=['GET'])
def get_students():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Group by student and get their statistics
        if not df_issues.empty:
            student_stats = df_issues.groupby('Member Code').agg({
                'Member Name': 'first',
                'Access No': 'count',
                'Return Date': lambda x: (x.isna()).sum(),
                'Issue Date': 'first'
            }).rename(columns={
                'Access No': 'total_issues',
                'Return Date': 'active_issues'
            }).reset_index().head(100)
            
            students = student_stats.to_dict(orient='records')
        else:
            students = []
        return jsonify({'success': True, 'data': students, 'total': len(students)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        dept_counts, date_counts, top_readers = get_admin_insights()
        stats = {
            'total_issues': len(df_issues),
            'unique_students': df_issues['Member Code'].nunique() if not df_issues.empty else 0,
            'unique_books': df_books['Title'].nunique() if not df_books.empty else 0,
            'total_books': len(df_books) if not df_books.empty else 0,
            'department_data': dept_counts,
            'date_data': date_counts,
            'top_readers': top_readers
        }
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/login_events/clear', methods=['POST'])
def clear_login_events():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        deleted = db.session.query(LoginEvent).delete()
        db.session.commit()
        return jsonify({'success': True, 'deleted': int(deleted or 0)})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/login-events', methods=['GET'])
def get_login_events():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        limit = request.args.get('limit', 20, type=int) or 20
        limit = max(1, min(limit, 200))

        rows = LoginEvent.query.order_by(LoginEvent.logged_in_at.desc()).limit(limit).all()
        data = []
        for ev in rows:
            data.append({
                "id": ev.id,
                "logged_in_at": ev.logged_in_at.strftime("%d-%m-%Y %H:%M") if ev.logged_in_at else "",
                "username": (ev.username or "").strip().lower(),
                "role": (ev.role or "").strip().lower(),
                "attempted_role": (ev.attempted_role or "").strip().lower(),
                "status": (ev.status or "success").strip().lower(),
                "ip_address": (ev.ip_address or "").strip(),
                "note": (ev.note or "").strip(),
            })
        return jsonify({'success': True, 'data': data, 'total': len(data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/active-users', methods=['GET'])
def get_active_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        limit = request.args.get('limit', 100, type=int) or 100
        limit = max(1, min(limit, 500))
        hours = request.args.get('hours', 24, type=int) or 24
        hours = max(1, min(hours, 720))
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        rows = (
            LoginEvent.query
            .filter(LoginEvent.status == 'success')
            .order_by(LoginEvent.logged_in_at.desc())
            .limit(5000)
            .all()
        )

        users_map = {}
        for ev in rows:
            username = (ev.username or "").strip().lower()
            if not username:
                continue
            if not ev.logged_in_at or ev.logged_in_at < cutoff:
                continue
            if username not in users_map:
                users_map[username] = {
                    "username": username,
                    "role": (ev.role or "").strip().lower(),
                    "last_login": ev.logged_in_at.strftime("%d-%m-%Y %H:%M"),
                    "ip_address": (ev.ip_address or "").strip(),
                    "login_count": 1,
                }
            else:
                users_map[username]["login_count"] += 1

        data = list(users_map.values())[:limit]
        return jsonify({'success': True, 'data': data, 'total': len(data), 'window_hours': hours})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/book-requests', methods=['GET'])
def get_book_requests():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    status = (request.args.get('status') or 'pending').strip().lower()
    q = BookRequest.query
    if status in ['pending', 'approved', 'rejected']:
        q = q.filter(BookRequest.status == status)

    try:
        rows = q.order_by(BookRequest.requested_at.desc()).limit(200).all()
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "username": (r.username or "").strip().lower(),
                "role": (r.role or "").strip().lower(),
                "email": (r.email or "").strip(),
                "book_title": r.book_title,
                "access_no": r.access_no or "",
                "category": r.category or "",
                "requested_due_date": r.requested_due_date or "",
                "status": r.status,
                "requested_at": r.requested_at.strftime("%d-%m-%Y %H:%M") if r.requested_at else "",
            })
        return jsonify({'success': True, 'data': data, 'total': len(data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _build_book_request_email(req: BookRequest, approved: bool, due_date: Optional[str] = None) -> tuple[str, str]:
    site_name = "SXC Smart Library"
    status_word = "Approved" if approved else "Rejected"
    subject = f"SXC Library | Book Request {status_word}"
    due_line = ""
    if approved and due_date:
        due_line = f"Due Date: {due_date}\n"
    acknowledgement = (
        "Your request is approved. Please collect the book from the library counter within the next 2 working days."
        if approved
        else "Sorry, your request could not be approved at this time. Please contact the library for details."
    )
    if approved and due_date:
        acknowledgement = f"{acknowledgement}\nPlease return the book on or before {due_date}."

    body = (
        f"Hello {req.username},\n\n"
        f"Your book request has been {status_word.lower()}.\n\n"
        f"Website: {site_name}\n"
        f"Book Title: {req.book_title}\n"
        f"Access No: {req.access_no or '-'}\n"
        f"Category: {req.category or '-'}\n"
        f"Requested On: {req.requested_at.strftime('%d-%m-%Y %H:%M') if req.requested_at else '-'}\n\n"
        f"{due_line}"
        f"Acknowledgement:\n{acknowledgement}\n\n"
        f"Regards,\nSXC Library Admin\n"
    )
    return subject, body


@app.route('/api/admin/book-requests/<int:req_id>/approve', methods=['POST'])
def approve_book_request(req_id: int):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    req = BookRequest.query.get(req_id)
    if not req:
        return jsonify({'success': False, 'error': 'Request not found'}), 404
    if (req.status or '').strip().lower() != 'pending':
        return jsonify({'success': False, 'error': 'Request already processed'}), 400

    payload = request.get_json(silent=True) or {}
    due_date = (payload.get('due_date') or '').strip()
    if not due_date:
        return jsonify({'success': False, 'error': 'Due date is required to approve.'}), 400
    try:
        due_date = datetime.strptime(due_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return jsonify({'success': False, 'error': 'Due date must be in YYYY-MM-DD format.'}), 400

    try:
        recipient_email = (req.email or "").strip()
        if not recipient_email:
            user = User.query.filter(db.func.lower(User.username) == (req.username or "").strip().lower()).first()
            recipient_email = (user.email or "").strip() if user else ""
            if recipient_email:
                req.email = recipient_email

        req.status = 'approved'
        req.reviewed_at = datetime.utcnow()
        req.reviewed_by = (session.get('user_id') or '').strip().lower()
        db.session.commit()

        email_sent = False
        email_reason = "email_missing"
        email_error = ""
        if recipient_email:
            subject, body = _build_book_request_email(req, approved=True, due_date=due_date)
            email_sent, email_reason, email_error = _send_email(recipient_email, subject, body)

        return jsonify({'success': True, 'email_sent': email_sent, 'email_reason': email_reason, 'email_error': email_error})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/book-requests/<int:req_id>/reject', methods=['POST'])
def reject_book_request(req_id: int):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    req = BookRequest.query.get(req_id)
    if not req:
        return jsonify({'success': False, 'error': 'Request not found'}), 404
    if (req.status or '').strip().lower() != 'pending':
        return jsonify({'success': False, 'error': 'Request already processed'}), 400

    payload = request.get_json(silent=True) or {}
    note = (payload.get('note') or '').strip() or None

    try:
        recipient_email = (req.email or "").strip()
        if not recipient_email:
            user = User.query.filter(db.func.lower(User.username) == (req.username or "").strip().lower()).first()
            recipient_email = (user.email or "").strip() if user else ""
            if recipient_email:
                req.email = recipient_email

        req.status = 'rejected'
        req.reviewed_at = datetime.utcnow()
        req.reviewed_by = (session.get('user_id') or '').strip().lower()
        req.note = note
        db.session.commit()

        email_sent = False
        email_reason = "email_missing"
        email_error = ""
        if recipient_email:
            subject, body = _build_book_request_email(req, approved=False)
            email_sent, email_reason, email_error = _send_email(recipient_email, subject, body)

        return jsonify({'success': True, 'email_sent': email_sent, 'email_reason': email_reason, 'email_error': email_error})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/search', methods=['POST'])
def admin_search():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        query = request.json.get('query', '').lower()
        search_type = request.json.get('type', 'all')  # all, books, transactions, students
        
        results = {}
        
        if search_type in ['all', 'books']:
            books_result = df_books[df_books['Title'].str.lower().str.contains(query, na=False)].head(20)
            results['books'] = books_result.to_dict(orient='records')
        
        if search_type in ['all', 'transactions']:
            trans_result = df_issues[
                (df_issues['Title'].str.lower().str.contains(query, na=False)) |
                (df_issues['Member Name'].str.lower().str.contains(query, na=False)) |
                (df_issues['Member Code'].str.lower().str.contains(query, na=False))
            ].head(20)
            results['transactions'] = trans_result.to_dict(orient='records')
        
        if search_type in ['all', 'students']:
            student_result = df_issues[df_issues['Member Name'].str.lower().str.contains(query, na=False)].head(20)
            results['students'] = student_result.to_dict(orient='records')
        
        return jsonify({'success': True, 'data': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/filter', methods=['POST'])
def admin_filter():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        payload = request.get_json(force=True) or {}
        dataset = payload.get('dataset', 'transactions')
        date_range = payload.get('dateRange', 'all')
        start_date = payload.get('startDate')
        end_date = payload.get('endDate')
        department = payload.get('department', 'all')
        category = payload.get('category', 'all')
        status = payload.get('status', 'all')
        search_query = payload.get('searchQuery', '').lower().strip()

        source = CSV_SOURCES.get(dataset)
        if not source:
            return jsonify({'success': False, 'error': f'Unknown dataset: {dataset}'}), 400

        def ensure_columns(df, cols):
            for col in cols:
                if col not in df.columns:
                    df[col] = ""
            return df

        def parse_issue_dates(df):
            for col in ['Issue Date', 'Due Date', 'Return Date']:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            return df

        def filter_issues(df, books_df):
            result = df.copy()
            result = ensure_columns(result, ['Title', 'Member Name', 'Member Code', 'Access No', 'Issue Date', 'Due Date', 'Return Date'])
            result = parse_issue_dates(result)

            if search_query:
                result = result[
                    (result['Title'].astype(str).str.lower().str.contains(search_query)) |
                    (result['Member Name'].astype(str).str.lower().str.contains(search_query)) |
                    (result['Member Code'].astype(str).str.lower().str.contains(search_query)) |
                    (result['Access No'].astype(str).str.contains(search_query))
                ]

            if department != 'all':
                result = result[result['Member Code'].astype(str).str.upper().str.contains(department.upper())]

            if category != 'all' and not books_df.empty:
                if 'Access No' in books_df.columns and 'Category' in books_df.columns:
                    access_no_in_cat = books_df[books_df['Category'].astype(str) == category]['Access No'].astype(str)
                    result = result[result['Access No'].astype(str).isin(access_no_in_cat)]

            now = pd.Timestamp.now().normalize()
            anchor = now
            try:
                if 'Issue Date' in result.columns and result['Issue Date'].notna().any():
                    anchor = pd.Timestamp(result['Issue Date'].max()).normalize()
            except Exception:
                anchor = now

            if date_range == 'today':
                result = result[result['Issue Date'].dt.date == anchor.date()]
            elif date_range == 'week':
                result = result[result['Issue Date'] >= (anchor - pd.Timedelta(days=7))]
            elif date_range == 'month':
                result = result[result['Issue Date'] >= (anchor - pd.Timedelta(days=30))]
            elif date_range == 'custom' and start_date and end_date:
                start_dt = pd.to_datetime(start_date, errors='coerce')
                end_dt = pd.to_datetime(end_date, errors='coerce')
                if pd.notna(start_dt) and pd.notna(end_dt):
                    result = result[(result['Issue Date'] >= start_dt) & (result['Issue Date'] <= end_dt)]

            if status == 'active':
                result = result[result['Return Date'].isna()]
            elif status == 'returned':
                result = result[result['Return Date'].notna()]
            elif status == 'overdue':
                result = result[(result['Return Date'].isna()) & (result['Due Date'] < now)]

            return result

        def build_stats(filtered_issues, books_df):
            now = pd.Timestamp.now().normalize()
            total_issues = len(filtered_issues)
            unique_students = filtered_issues['Member Code'].nunique() if 'Member Code' in filtered_issues.columns else 0
            # Include approved portal users so admin-approved accounts are reflected in dashboard counts.
            portal_users_count = 0
            try:
                users_q = User.query.filter(User.role.in_(['student', 'staff']))
                if search_query:
                    search_term = f"%{search_query}%"
                    users_q = users_q.filter(
                        (db.func.lower(User.username).like(search_term))
                        | (db.func.lower(db.func.coalesce(User.full_name, '')).like(search_term))
                        | (db.func.lower(db.func.coalesce(User.email, '')).like(search_term))
                    )
                if department != 'all':
                    users_q = users_q.filter(db.func.lower(User.username).like(f"%{department.lower()}%"))
                portal_users_count = int(users_q.count())
            except Exception:
                portal_users_count = 0
            unique_students = max(int(unique_students), int(portal_users_count))

            inventory_unique_titles = 0
            if not books_df.empty and 'Title' in books_df.columns:
                inventory_unique_titles = int(books_df['Title'].astype(str).nunique())
            active_issues = 0
            returned_issues = 0
            return_rate = 0.0
            latest_issue_date = ""
            try:
                if 'Return Date' in filtered_issues.columns:
                    active_issues = int(filtered_issues['Return Date'].isna().sum())
                    returned_issues = int(filtered_issues['Return Date'].notna().sum())
                    return_rate = round((returned_issues / total_issues) * 100, 1) if total_issues else 0.0
                if 'Issue Date' in filtered_issues.columns and filtered_issues['Issue Date'].notna().any():
                    latest_issue_date = filtered_issues['Issue Date'].max().strftime('%d-%m-%Y')
            except Exception:
                active_issues = 0
                returned_issues = 0
                return_rate = 0.0
                latest_issue_date = ""
            overdue_count = 0
            if 'Return Date' in filtered_issues.columns and 'Due Date' in filtered_issues.columns:
                overdue_count = len(filtered_issues[(filtered_issues['Return Date'].isna()) & (filtered_issues['Due Date'] < now)])
            return {
                'total_issues': total_issues,
                'unique_students': unique_students,
                'unique_books': inventory_unique_titles,
                'overdue_count': overdue_count,
                'active_issues': active_issues,
                'returned_issues': returned_issues,
                'return_rate': return_rate,
                'latest_issue_date': latest_issue_date,
                'total_books': len(books_df) if not books_df.empty else 0
            }

        def build_charts(filtered_issues, books_df):
            def extract_dept(code):
                match = re.search(r'([a-zA-Z]+)', str(code))
                return match.group(1).upper() if match else 'OTHER'

            charts = {}
            if filtered_issues.empty:
                charts['dept'] = {'labels': [], 'values': []}
                charts['month'] = {'labels': [], 'values': []}
                charts['heatmap'] = {'labels': [], 'issued': [], 'returned': []}
                charts['radar'] = {'labels': [], 'values': []}
                charts['status'] = {'labels': ['Avail', 'Issued', 'Overdue'], 'values': [0, 0, 0]}
                charts['transaction'] = {'labels': ['Returned', 'Active'], 'values': [0, 0]}
                charts['category'] = {'labels': [], 'values': []}
                return charts

            issues = filtered_issues.copy()
            issues['Dept'] = issues['Member Code'].apply(extract_dept)
            dept_counts = issues['Dept'].value_counts().head(5)
            charts['dept'] = {'labels': dept_counts.index.tolist(), 'values': dept_counts.values.tolist()}
            charts['radar'] = {'labels': dept_counts.index.tolist(), 'values': dept_counts.values.tolist()}

            # Anchor charts to the latest date present in data (so visuals work even if CSV is old)
            anchor = pd.Timestamp.now().normalize()
            try:
                if 'Issue Date' in issues.columns and issues['Issue Date'].notna().any():
                    anchor = issues['Issue Date'].max()
                elif 'Return Date' in issues.columns and issues['Return Date'].notna().any():
                    anchor = issues['Return Date'].max()
            except Exception:
                anchor = pd.Timestamp.now().normalize()
            anchor = pd.Timestamp(anchor).normalize()

            # Monthly trend (last 6 months from anchor)
            if 'Issue Date' in issues.columns and issues['Issue Date'].notna().any():
                month_counts = issues['Issue Date'].dropna().dt.to_period('M').value_counts().sort_index()
                anchor_month = anchor.to_period('M')
                months = [anchor_month - i for i in range(5, -1, -1)]
                charts['month'] = {
                    'labels': [m.to_timestamp().strftime('%b-%y') for m in months],
                    'values': [int(month_counts.get(m, 0)) for m in months],
                }
            else:
                charts['month'] = {'labels': [], 'values': []}

            now = pd.Timestamp.now().normalize()
            week_labels = []
            issued_values = []
            returned_values = []
            # Last 4 weeks from anchor (inclusive)
            week_ranges = []
            end = anchor
            for _ in range(4):
                start = end - pd.Timedelta(days=6)
                week_ranges.append((start, end))
                end = start - pd.Timedelta(days=1)
            week_ranges.reverse()

            for start, end in week_ranges:
                week_labels.append(f"{start.strftime('%d-%m')} - {end.strftime('%d-%m')}")
                issued_values.append(len(issues[(issues['Issue Date'] >= start) & (issues['Issue Date'] <= end)]))
                returned_values.append(len(issues[(issues['Return Date'].notna()) & (issues['Return Date'] >= start) & (issues['Return Date'] <= end)]))
            charts['heatmap'] = {'labels': week_labels, 'issued': issued_values, 'returned': returned_values}

            total_books = len(books_df) if not books_df.empty else 0
            active_issues = len(issues[issues['Return Date'].isna()])
            overdue_issues = len(issues[(issues['Return Date'].isna()) & (issues['Due Date'] < now)])
            available = max(total_books - active_issues, 0)
            charts['status'] = {'labels': ['Avail', 'Issued', 'Overdue'], 'values': [available, active_issues, overdue_issues]}

            returned = len(issues[issues['Return Date'].notna()])
            charts['transaction'] = {'labels': ['Returned', 'Active'], 'values': [returned, active_issues]}

            if not books_df.empty and 'Category' in books_df.columns and 'Access No' in books_df.columns:
                access_in_issues = issues['Access No'].astype(str).unique().tolist()
                books_filtered = books_df[books_df['Access No'].astype(str).isin(access_in_issues)]
                cat_counts = books_filtered['Category'].astype(str).value_counts().head(5)
                charts['category'] = {'labels': cat_counts.index.tolist(), 'values': cat_counts.values.tolist()}
            else:
                charts['category'] = {'labels': [], 'values': []}

            return charts

        # Read CSV fresh on each filter request
        if dataset in ['transactions', 'students']:
            issues_df = pd.read_csv(CSV_SOURCES['transactions'])
        else:
            issues_df = pd.DataFrame()

        books_df = pd.DataFrame()
        try:
            books_df = pd.read_csv(CSV_SOURCES['books'])
        except Exception:
            books_df = pd.DataFrame()

        if dataset == 'books':
            result = pd.read_csv(source)
            result = ensure_columns(result, ['Title', 'Category', 'Access No'])
            if search_query:
                result = result[
                    (result['Title'].astype(str).str.lower().str.contains(search_query)) |
                    (result['Category'].astype(str).str.lower().str.contains(search_query)) |
                    (result['Access No'].astype(str).str.contains(search_query))
                ]
            if category != 'all':
                result = result[result['Category'].astype(str) == category]
            filtered_data = result.fillna("").head(200).to_dict(orient='records')
            return jsonify({'success': True, 'data': filtered_data, 'total': len(result)})

        if dataset == 'students':
            issues_filtered = filter_issues(issues_df, books_df)
            if issues_filtered.empty:
                return jsonify({'success': True, 'data': [], 'total': 0})

            student_stats = issues_filtered.groupby('Member Code', dropna=False).agg({
                'Member Name': 'first',
                'Access No': 'count',
                'Return Date': lambda x: x.isna().sum()
            }).rename(columns={
                'Access No': 'total_issues',
                'Return Date': 'active_issues'
            }).reset_index()

            students = student_stats.fillna("").head(200).to_dict(orient='records')
            return jsonify({'success': True, 'data': students, 'total': len(student_stats)})

        # Default: transactions
        issues_filtered = filter_issues(issues_df, books_df)
        stats = build_stats(issues_filtered, books_df)
        charts = build_charts(issues_filtered, books_df)

        issues_display = issues_filtered.copy()
        now = pd.Timestamp.now().normalize()
        issues_display['Status'] = 'Active'
        issues_display.loc[issues_display['Return Date'].notna(), 'Status'] = 'Returned'
        issues_display.loc[
            issues_display['Return Date'].isna()
            & issues_display['Due Date'].notna()
            & (issues_display['Due Date'] < now),
            'Status',
        ] = 'Overdue'

        ref_date = issues_display['Return Date'].fillna(now)
        late_days = (ref_date.dt.normalize() - issues_display['Due Date'].dt.normalize()).dt.days
        issues_display['Fine'] = late_days.where(late_days > 0, 0).fillna(0).astype(int)

        issues_display = issues_display.sort_values(['Issue Date'], ascending=[False], na_position='last')
        issues_display['Issue Date'] = issues_display['Issue Date'].dt.strftime('%d-%m-%Y').fillna("")
        issues_display['Due Date'] = issues_display['Due Date'].dt.strftime('%d-%m-%Y').fillna("")
        issues_display['Return Date'] = issues_display['Return Date'].dt.strftime('%d-%m-%Y').fillna("")

        filtered_data = issues_display.fillna("").head(200).to_dict(orient='records')

        return jsonify({
            'success': True,
            'data': filtered_data,
            'total': len(issues_filtered),
            'stats': stats,
            'charts': charts
        })

    except Exception as e:
        print(f"Filter Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
# PDF Report Generation
class PDFReport(FPDF):
    def __init__(
        self,
        report_title: str,
        report_subtitle: str,
        generated_at: str,
        summary_line: str,
        columns: list[str],
        col_widths: list[int],
    ):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.report_title = report_title
        self.report_subtitle = report_subtitle
        self.generated_at = generated_at
        self.summary_line = summary_line
        self.columns = columns
        self.col_widths = col_widths

        # Layout
        self.set_margins(7, 10, 7)
        self.set_auto_page_break(auto=True, margin=12)

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 7, self.report_title, 0, 1, "C")
        self.set_font("Arial", "", 11)
        self.cell(0, 6, self.report_subtitle, 0, 1, "C")
        self.set_font("Arial", "", 9)
        meta = f"Generated: {self.generated_at}"
        if self.summary_line:
            meta = f"{meta} | {self.summary_line}"
        self.cell(0, 5, meta, 0, 1, "C")
        self.ln(2)

        # Table header
        self.set_fill_color(230, 235, 245)
        self.set_font("Arial", "B", 9)
        for col, w in zip(self.columns, self.col_widths):
            self.cell(int(w), 7, str(col), 1, 0, "C", fill=True)
        self.ln()
        self.set_font("Arial", "", 8)

    def footer(self):
        self.set_y(-10)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


@app.route("/admin/download_report")
def download_pdf():
    if "user_id" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))

    try:
        issues = df_issues.copy() if not df_issues.empty else pd.DataFrame()
        for col in ["Member Code", "Member Name", "Access No", "Title", "Issue Date", "Due Date", "Return Date"]:
            if col not in issues.columns:
                issues[col] = ""

        issue_dt = pd.to_datetime(issues["Issue Date"], dayfirst=True, errors="coerce")
        due_dt = pd.to_datetime(issues["Due Date"], dayfirst=True, errors="coerce")
        ret_dt = pd.to_datetime(issues["Return Date"], dayfirst=True, errors="coerce")
        now = pd.Timestamp.now().normalize()

        issues["_issue_dt"] = issue_dt
        issues["_due_dt"] = due_dt
        issues["_ret_dt"] = ret_dt

        issues["Status"] = "Active"
        issues.loc[issues["_ret_dt"].notna(), "Status"] = "Returned"
        issues.loc[issues["_ret_dt"].isna() & issues["_due_dt"].notna() & (issues["_due_dt"] < now), "Status"] = "Overdue"

        total = int(len(issues))
        returned = int((issues["Status"] == "Returned").sum()) if total else 0
        overdue = int((issues["Status"] == "Overdue").sum()) if total else 0
        active = int((issues["Status"] == "Active").sum()) if total else 0
        summary_line = f"Total: {total} | Active: {active} | Returned: {returned} | Overdue: {overdue}"

        # Latest first; keep export bounded
        issues = issues.sort_values(["_issue_dt"], ascending=[False], na_position="last").fillna("").head(200)

        # Table layout
        page_width = 297  # A4 landscape width in mm
        left_margin = 7
        right_margin = 7
        available = page_width - left_margin - right_margin

        columns = ["Issue", "Due", "Return", "Status", "Member Code", "Member Name", "Acc No", "Title"]
        fixed = [23, 23, 23, 20, 26, 50, 20]
        title_width = max(int(available - sum(fixed)), 60)
        col_widths = fixed + [title_width]

        generated_at = datetime.now().strftime("%d-%m-%Y %H:%M")
        pdf = PDFReport(
            report_title="St. Xavier's College (Autonomous), Palayamkottai",
            report_subtitle="Library Transactions Report",
            generated_at=generated_at,
            summary_line=summary_line,
            columns=columns,
            col_widths=col_widths,
        )
        pdf.add_page()

        def fmt_date(value) -> str:
            if value is None:
                return "-"
            try:
                ts = pd.to_datetime(value, dayfirst=True, errors="coerce")
                if pd.isna(ts):
                    return "-"
                return pd.Timestamp(ts).strftime("%d-%m-%Y")
            except Exception:
                return "-"

        def fit_text(text: str, width: int) -> str:
            s = re.sub(r"\\s+", " ", str(text or "")).strip()
            # FPDF (classic) is Latin-1 based; replace unsupported chars safely.
            try:
                s = s.encode("latin-1", errors="replace").decode("latin-1")
            except Exception:
                pass
            if not s:
                return ""
            if pdf.get_string_width(s) <= (width - 2):
                return s

            ell = "..."
            max_w = max((width - 2) - pdf.get_string_width(ell), 0)
            out = ""
            for ch in s:
                if pdf.get_string_width(out + ch) > max_w:
                    break
                out += ch
            return (out + ell) if out else (s[:1] + ell)

        # Rows
        row_h = 6
        zebra = False
        for _, row in issues.iterrows():
            zebra = not zebra
            fill = zebra
            pdf.set_fill_color(248, 249, 250) if fill else pdf.set_fill_color(255, 255, 255)

            status = str(row.get("Status", "") or "").strip() or "Active"
            cells = [
                fmt_date(row.get("_issue_dt")),
                fmt_date(row.get("_due_dt")),
                fmt_date(row.get("_ret_dt")),
                status,
                str(row.get("Member Code", "") or "").strip(),
                str(row.get("Member Name", "") or "").strip(),
                str(row.get("Access No", "") or "").strip(),
                str(row.get("Title", "") or "").strip(),
            ]

            aligns = ["L", "L", "L", "C", "L", "L", "L", "L"]
            for w, value, align in zip(col_widths, cells, aligns):
                pdf.cell(int(w), row_h, fit_text(value, int(w)), 1, 0, align, fill=fill)
            pdf.ln(row_h)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        filename = f"SXC_Library_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf"
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    ensure_app_bootstrapped()
    host = (os.getenv("HOST") or "0.0.0.0").strip()
    port = int((os.getenv("PORT") or "5000").strip())
    debug = _env_bool("FLASK_DEBUG", default=False)
    app.run(host=host, port=port, debug=debug)
