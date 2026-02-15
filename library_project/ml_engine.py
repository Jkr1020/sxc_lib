import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import difflib
import re
from datetime import datetime

# ============================================================================
# 📚 LIBBOT - OFFICIAL LIBRARY AI ASSISTANT SYSTEM PROMPTS
# St. Xavier's College (Autonomous), Palayamkottai
# ============================================================================

MASTER_SYSTEM_PROMPT = """
You are "LibBot", the official AI-powered Library Assistant of
St. Xavier's College (Autonomous), Palayamkottai.

You operate as a professional academic library assistant for both
STUDENTS and ADMIN users.

You are trained strictly on:
1. Library Book Inventory Dataset
2. Student Issue & Return Transaction Dataset
3. Official Library Rules, Timings, and Policies

Your primary goals are:
• Help students find books and check availability
• Answer questions about issue status, due dates, and overdue books
• Recommend books based on department and subject
• Assist admins with library statistics and analytics
• Summarize real-time dashboard data when requested

You must always give clear, concise, and factual answers.
Use formal academic English.
"""

DATA_CONSTRAINT_PROMPT = """
IMPORTANT DATA RULES:

• Respond ONLY using information available in the datasets
• NEVER create or guess book titles, authors, or student data
• If data is unavailable, respond:
  "The requested information is not available in the library records."

• Do NOT answer questions unrelated to the library system
• Do NOT provide personal opinions
"""

DATA_STRUCTURE_PROMPT = """
LIBRARY BOOK INVENTORY DATASET:
Columns include:
- Access No
- Title
- Author
- Category
- Department
- Total Copies
- Available Copies

STUDENT ISSUE & RETURN DATASET:
Columns include:
- Student ID
- Student Name
- Department
- Access No
- Issue Date
- Due Date
- Return Date
- Status (Active / Returned / Overdue)
"""

ROLE_BASED_BEHAVIOR_PROMPT = """
USER ROLE HANDLING:

If user_role = "student":
• Answer book search queries
• Show issue status for that student
• Display due dates and overdue alerts
• Recommend books by department or subject

If user_role = "admin":
• Provide dashboard statistics
• Show department-wise usage
• Identify top readers
• Summarize trends (monthly issues, overdues)
• Assist in decision-making (stock demand insights)
"""

RESPONSE_STYLE_PROMPT = """
RESPONSE GUIDELINES:

• Use polite and professional language
• Address users as "Student" or "Admin"
• Keep responses short and informative
• Use bullet points for lists
• Use tables when summarizing analytics
"""

FALLBACK_PROMPT = """
If a user asks a question outside library scope, reply:

"I am designed to assist only with St. Xavier's College Library-related queries."
"""

# ============================================================================
# 🤖 LIBBOT CORE ENGINE
# ============================================================================

class LibraryBrain:
    def __init__(self, book_source, issue_source=None):
        print("[INFO] Initializing Professional Library AI Engine...")
        try:
            # 1. Load Books
            self.raw_df = pd.read_csv(book_source)
            self.raw_df['Title'] = self.raw_df['Title'].fillna('Unknown Title')
            self.raw_df['Category'] = self.raw_df['Category'].fillna('General')
            self.raw_df['Title_Clean'] = self.raw_df['Title'].astype(str).str.lower().str.strip()
            
            # 2. Build Knowledge Base from Books
            self.knowledge_base = {
                'total_books': len(self.raw_df),
                'categories': self.raw_df['Category'].unique().tolist(),
                'category_counts': self.raw_df['Category'].value_counts().to_dict()
            }
            
            # 3. Process Transaction Data for Popularity & Analytics
            self.popularity_map = {}
            self.transaction_insights = {}
            self.student_stats = {}
            
            if issue_source:
                try:
                    df_issues = pd.read_csv(issue_source)
                    df_issues['Title_Clean'] = df_issues['Title'].astype(str).str.lower().str.strip()
                    
                    # Popularity scoring
                    pop_counts = df_issues['Title_Clean'].value_counts()
                    max_count = pop_counts.max() if len(pop_counts) > 0 else 1
                    self.popularity_map = (pop_counts / max_count).to_dict()
                    
                    # Top borrowed books
                    self.knowledge_base['top_books'] = pop_counts.head(20).to_dict()
                    self.knowledge_base['total_transactions'] = len(df_issues)
                    self.knowledge_base['unique_students'] = df_issues['Member Code'].nunique()
                    
                    # Department insights
                    depts = {}
                    for code in df_issues['Member Code']:
                        dept_match = re.search(r'([a-zA-Z]+)', str(code))
                        if dept_match:
                            dept = dept_match.group(1).upper()
                            depts[dept] = depts.get(dept, 0) + 1
                    self.knowledge_base['departments'] = depts
                    
                    print(f"[INFO] Loaded {len(self.popularity_map)} popular titles from transactions")
                    print(f"[INFO] {self.knowledge_base['unique_students']} students, {self.knowledge_base['total_transactions']} transactions")
                except Exception as e:
                    print(f"[WARN] Issue loading transaction data: {e}")

            # 4. Prepare DataFrame with unique titles
            self.df = self.raw_df.drop_duplicates(subset='Title_Clean').copy()
            self.df = self.df.reset_index(drop=True)
            self.df['pop_score'] = self.df['Title_Clean'].map(self.popularity_map).fillna(0)
            self.df['Category_Clean'] = self.df['Category'].astype(str).str.lower().str.strip()
            
            # 5. Enhanced Content for semantic search
            self.df['content'] = (
                self.df['Title'].astype(str).str.lower() + " " + 
                self.df['Category'].astype(str).str.lower()
            )
            
            # 6. TF-IDF Vectorization with better parameters
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                min_df=1,
                max_features=5000,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['content'])
            self.indices = pd.Series(self.df.index, index=self.df['Title_Clean']).drop_duplicates()
            
            print(f"[INFO] Professional Library AI Ready - {len(self.df)} unique titles indexed")
            
        except Exception as e:
            print(f"[ERROR] AI Engine Initialization Failed: {e}")
            self.df = pd.DataFrame()
            self.knowledge_base = {}


    def _resolve_title_key(self, title: str):
        """Resolve a user-provided title to an internal Title_Clean key."""
        clean_title = str(title or "").lower().strip()
        if not clean_title:
            return None

        if clean_title in self.indices:
            return clean_title

        try:
            keys = list(self.indices.index.astype(str))
        except Exception:
            keys = []

        # Fast contains match (useful for "part-2", etc.)
        for k in keys[:2000]:
            if clean_title and clean_title in k:
                return k

        # Fuzzy match fallback
        matches = difflib.get_close_matches(clean_title, keys, n=1, cutoff=0.65)
        if matches:
            return matches[0]

        return None


    def _resolve_category_from_text(self, text: str):
        """Resolve a user query to the closest known category (if any)."""
        categories = self.knowledge_base.get('categories', []) or []
        categories = [str(c).strip() for c in categories if str(c).strip()]
        if not categories:
            return None

        text_lower = str(text or "").lower()

        # Direct substring match (most reliable)
        for cat in categories:
            if cat.lower() in text_lower:
                return cat

        # Fuzzy match on a cleaned query (useful for "maths" -> "mathematics")
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", text_lower)
        cleaned = re.sub(
            r"\b(recommend|suggest|books?|book|category|categories|in|on|for|please|me|show|top|popular|trending|best|give|want|need|list)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None

        cat_map = {c.lower(): c for c in categories}
        matches = difflib.get_close_matches(cleaned, list(cat_map.keys()), n=1, cutoff=0.74)
        if matches:
            return cat_map[matches[0]]

        return None


    def recommend_by_category(self, category: str, n: int = 5):
        """Recommend books within a category (popularity-first)."""
        if self.df.empty:
            return []

        category = str(category or "").strip()
        if not category:
            return []

        cat_clean = category.lower().strip()
        pool = self.df[self.df['Category_Clean'] == cat_clean].copy()
        if pool.empty:
            return []

        pool = pool.sort_values(['pop_score', 'Title_Clean'], ascending=[False, True])

        results = []
        for _, row in pool.head(max(int(n), 0)).iterrows():
            results.append({
                'title': row['Title'],
                'category': row['Category'],
                'relevance': round(float(row.get('pop_score', 0)) * 100, 1)
            })
        return results


    def get_recommendations(self, title, n=5):
        """Category-aware content-based recommendations with hybrid scoring.

        Prevents irrelevant cross-category results by prioritizing the same
        category as the seed title. Falls back to global similarity only when
        category matches are unavailable.
        """
        if self.df.empty:
            return []
        
        try:
            key = self._resolve_title_key(title)
            if not key:
                return []

            idx = self.indices[key]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            idx = int(idx)

            cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

            seed_category = str(self.df.at[idx, 'Category_Clean'] or "").strip()
            if seed_category:
                candidate_idx = self.df.index[self.df['Category_Clean'] == seed_category].to_numpy(dtype=int)
                candidate_idx = candidate_idx[candidate_idx != idx]
                min_sim = 0.0
            else:
                candidate_idx = np.array([i for i in range(len(self.df)) if i != idx], dtype=int)
                # For global fallback, require some semantic overlap.
                min_sim = 0.08

            if candidate_idx.size == 0:
                return []

            sims = cosine_sim[candidate_idx]
            pops = self.df.loc[candidate_idx, 'pop_score'].to_numpy(dtype=float)

            # Hybrid scoring: 85% semantic similarity + 15% popularity
            hybrid = (sims * 0.85) + (pops * 0.15)

            if min_sim > 0:
                keep = sims >= min_sim
                candidate_idx = candidate_idx[keep]
                sims = sims[keep]
                pops = pops[keep]
                hybrid = hybrid[keep]

            if candidate_idx.size == 0:
                return []

            order = np.argsort(hybrid)[::-1]
            order = order[: max(int(n), 0)]

            results = []
            used_idx = set([idx])
            for pos in order:
                cand_i = int(candidate_idx[pos])
                used_idx.add(cand_i)
                row = self.df.iloc[cand_i]
                results.append({
                    'title': row['Title'],
                    'category': row['Category'],
                    'relevance': round(float(hybrid[pos]) * 100, 1)
                })

            # If we scoped to a category but still got too few, fill from the
            # same category by popularity (keeps recommendations consistent).
            if seed_category and len(results) < n:
                remaining = n - len(results)
                cat_pool = self.df[self.df['Category_Clean'] == seed_category].copy()
                cat_pool = cat_pool[~cat_pool.index.isin(used_idx)]
                if not cat_pool.empty:
                    cat_pool = cat_pool.sort_values(['pop_score', 'Title_Clean'], ascending=[False, True])
                    for _, row in cat_pool.head(remaining).iterrows():
                        results.append({
                            'title': row['Title'],
                            'category': row['Category'],
                            'relevance': round(float(row.get('pop_score', 0)) * 100, 1)
                        })
            return results
            
        except Exception as e:
            print(f"[WARN] Recommendation Error: {e}")
            return []

    def semantic_search(self, query, n=5):
        """Advanced semantic search with relevance scoring and professional filtering."""
        if self.df.empty:
            return []
        
        try:
            query_lower = query.lower().strip()
            
            # Enhanced query vectorization
            query_vec = self.tfidf.transform([query_lower])
            cosine_sim = linear_kernel(query_vec, self.tfidf_matrix).flatten()
            
            # Filter and sort by relevance
            scored_results = []
            for i, score in enumerate(cosine_sim):
                if score > 0.05:  # Relevance threshold
                    row = self.df.iloc[i]
                    scored_results.append({
                        'index': i,
                        'score': score,
                        'title': row['Title'],
                        'category': row['Category'],
                        'access_no': row.get('Access No', 'N/A'),
                        'popularity': row.get('pop_score', 0)
                    })
            
            # Sort by combined relevance score
            scored_results.sort(key=lambda x: (x['score'] * 0.7 + x['popularity'] * 0.3), reverse=True)
            
            # Format for API
            results = []
            for item in scored_results[:n]:
                results.append({
                    'title': item['title'],
                    'category': item['category'],
                    'access_no': item['access_no'],
                    'relevance': round(item['score'] * 100, 1)
                })
            
            return results
            
        except Exception as e:
            print(f"[WARN] Search Error: {e}")
            return []
    
    def get_category_info(self, category=None):
        """Retrieve comprehensive category information."""
        if self.df.empty:
            return {}
        
        try:
            if category:
                books = self.df[self.df['Category'].str.lower() == category.lower()]
                return {
                    'category': category,
                    'total_books': len(books),
                    'books': books['Title'].head(10).tolist()
                }
            else:
                # All categories with counts
                return self.knowledge_base.get('category_counts', {})
        except Exception as e:
            print(f"[WARN] Category Info Error: {e}")
            return {}
    
    def get_library_stats(self):
        """Return comprehensive library statistics."""
        return {
            'total_books': self.knowledge_base.get('total_books', 0),
            'unique_titles': len(self.df),
            'total_categories': len(self.knowledge_base.get('categories', [])),
            'total_transactions': self.knowledge_base.get('total_transactions', 0),
            'active_students': self.knowledge_base.get('unique_students', 0),
            'departments': self.knowledge_base.get('departments', {}),
            'top_books': list(self.knowledge_base.get('top_books', {}).items())[:10]
        }
    
    def answer_question(self, question, user_id='Guest', user_role='guest', last_book=None):
        """Professional RAG-based natural language response generator using system prompts."""
        question_lower = str(question or "").lower().strip()
        
        # Apply MASTER_SYSTEM_PROMPT principles
        # Always respond factually based on data, never hallucinate
        
        # === GREETING DETECTION ===
        if (
            any(phrase in question_lower for phrase in ['good morning', 'good afternoon', 'good evening'])
            or re.search(r"\b(hello|hi|hey|greetings)\b", question_lower)
        ):
            return f"👋 **Welcome to LibBot!** I'm the official AI Library Assistant for St. Xavier's College.\n\nI can help you with:\n• 🔍 **Book Search** - Find any book in our library\n• 📊 **Library Statistics** - View borrowing trends and data\n• ⭐ **Recommendations** - Get personalized book suggestions\n• 📚 **Category Info** - Explore our collection\n• 👥 **Department Stats** - See department-wise analytics\n\nWhat would you like to know?"
        
        matched_category = self._resolve_category_from_text(question_lower)

        # === CATEGORY-SPECIFIC COUNTS / LISTS ===
        if matched_category and any(w in question_lower for w in ['how many', 'count', 'total']):
            if any(w in question_lower for w in ['book', 'books', 'title', 'titles']):
                info = self.get_category_info(matched_category)
                total = info.get('total_books', 0)
                return f"**{matched_category}** category has **{total}** book(s) in the library inventory."

        if matched_category and any(w in question_lower for w in ['list', 'show', 'give']):
            if any(w in question_lower for w in ['book', 'books', 'titles']):
                info = self.get_category_info(matched_category)
                books = info.get('books', []) or []
                if books:
                    response = f"**{matched_category}** - Sample Titles:\n\n"
                    for i, t in enumerate(books[:10], 1):
                        response += f"{i}. **{t}**\n"
                    response += "\nAsk: 'Recommend similar books' for more suggestions."
                    return response.strip()

        # === RECOMMENDATIONS (ADVANCED) ===
        if any(w in question_lower for w in ['recommend', 'suggest', 'similar', 'related', 'another', 'more']):
            short_followups = {'more', 'another', 'next', 'show more', 'more please'}
            if question_lower in short_followups and last_book:
                recs = self.get_recommendations(last_book, n=5)
                if recs:
                    response = f"**More recommendations based on:** **{last_book}**\n\n"
                    for i, r in enumerate(recs, 1):
                        response += f"{i}. **{r['title']}**  ({r['category']})\n"
                    return response.strip()

            seed = None
            m = re.search(r"(?:like|similar to|based on)\s+(.+)$", question_lower)
            if m:
                seed = m.group(1).strip()
            else:
                m = re.search(r"(?:recommend|suggest)\s+(?:me\s+)?(.+)$", question_lower)
                if m:
                    seed = m.group(1).strip()

            if seed:
                seed = re.sub(r"\b(books?|please)\b", "", seed).strip(" .!?\"'`")
                seed = re.sub(r"\s+", " ", seed).strip()

            # 1) Book-based recommendations
            if seed:
                display_seed = seed
                key = self._resolve_title_key(seed)
                if key:
                    seed_idx = self.indices[key]
                    if isinstance(seed_idx, pd.Series):
                        seed_idx = seed_idx.iloc[0]
                    try:
                        display_seed = str(self.df.iloc[int(seed_idx)]['Title'])
                    except Exception:
                        display_seed = seed

                recs = self.get_recommendations(seed, n=5)
                if recs:
                    response = f"**Recommended books based on:** **{display_seed}**\n\n"
                    for i, r in enumerate(recs, 1):
                        response += f"{i}. **{r['title']}**  ({r['category']})\n"
                    return response.strip()

            # 2) Category-based recommendations
            if matched_category:
                recs = self.recommend_by_category(matched_category, n=5)
                if recs:
                    response = f"**Recommended books in {matched_category}:**\n\n"
                    for i, r in enumerate(recs, 1):
                        response += f"{i}. **{r['title']}**\n"
                    return response.strip()

            # 3) Context-based fallback
            if last_book:
                recs = self.get_recommendations(last_book, n=5)
                if recs:
                    response = f"**Recommended books based on:** **{last_book}**\n\n"
                    for i, r in enumerate(recs, 1):
                        response += f"{i}. **{r['title']}**  ({r['category']})\n"
                    return response.strip()

            return "The requested recommendation is not available in the library records."

        # === LIBRARY STATISTICS ===
        if any(word in question_lower for word in ['statistics', 'stats', 'total', 'how many', 'count', 'overview']):
            stats = self.get_library_stats()
            response = f"""
📊 **SXC LIBRARY - COMPREHENSIVE STATISTICS**

**Collection Overview:**
• Total Books in Inventory: **{stats['total_books']}**
• Unique Titles: **{stats['unique_titles']}**
• Subject Categories: **{stats['total_categories']}**

**Usage Analytics:**
• Total Transactions (Issues): **{stats['total_transactions']:,}**
• Active Student Users: **{stats['active_students']}**

**Department Distribution:**
"""
            if stats['departments']:
                dept_list = sorted(stats['departments'].items(), key=lambda x: x[1], reverse=True)
                for dept, count in dept_list:
                    pct = round((count / sum(d[1] for d in dept_list)) * 100, 1)
                    response += f"  • {dept}: **{count}** transactions ({pct}%)\n"
            
            if stats['top_books']:
                response += f"\n**Most Borrowed Books:**\n"
                for idx, (book, count) in enumerate(stats['top_books'][:5], 1):
                    response += f"  {idx}. **{book}** - {int(count)} borrows\n"
            
            response += "\n*Data sourced from real library transaction records*"
            return response.strip()
        
        # === CATEGORY QUESTIONS ===
        if any(word in question_lower for word in ['category', 'categories', 'subject', 'subjects', 'genre', 'genres']):
            if matched_category:
                info = self.get_category_info(matched_category)
                books = info.get('books', []) or []

                response = f"**{matched_category}**\n\n"
                response += f"Total books: **{info.get('total_books', 0)}**\n"
                if books:
                    response += "\nSample titles:\n"
                    for t in books[:10]:
                        response += f"- **{t}**\n"
                return response.strip()

            categories = self.knowledge_base.get('categories', [])
            response = f"📚 **Available Book Categories ({len(categories)} total):**\n\n"
            cat_counts = self.knowledge_base.get('category_counts', {})
            for cat in sorted(categories):
                count = cat_counts.get(cat, 0)
                response += f"• **{cat}** ({count} books)\n"
            return response.strip()
        
        # === POPULAR/TOP BOOKS ===
        if any(word in question_lower for word in ['popular', 'most borrowed', 'trending', 'bestsell', 'most issued', 'top books', 'demand', 'highest demand']):
            top_books = self.knowledge_base.get('top_books', {})
            if top_books:
                response = "⭐ **MOST BORROWED BOOKS - SXC LIBRARY**\n\n"
                response += "Books with highest circulation and demand:\n\n"
                total_borrows = sum(top_books.values())
                for idx, (book, count) in enumerate(list(top_books.items())[:10], 1):
                    count = int(count)
                    pct = round((count / total_borrows) * 100, 1)
                    response += f"{idx:2}. **{book}**\n"
                    response += f"    → {count} issues | {pct}% of total demand\n\n"
                response += "*Based on actual library transaction records*"
                return response.strip()
            return "Currently no data on popular books."
        
        # === DEPARTMENT INFORMATION ===
        if any(word in question_lower for word in ['department', 'dept', 'departments', 'which department', 'usage by department', 'ucs', 'uma', 'uen', 'uba']):
            depts = self.knowledge_base.get('departments', {})
            if depts:
                total_students = sum(depts.values())
                response = "🏢 **DEPARTMENT-WISE LIBRARY USAGE**\n\n"
                response += "Student borrowing activity by department:\n\n"
                for dept, count in sorted(depts.items(), key=lambda x: x[1], reverse=True):
                    percentage = round((count / total_students) * 100, 1)
                    bar_length = int(percentage / 5)
                    bar = "█" * bar_length
                    response += f"**{dept:15}** | {bar:20} {count:4} students ({percentage:5.1f}%)\n"
                response += f"\nTotal Active Users: {total_students}"
                return response.strip()
            return "Data not available: Department information not loaded."
        
        # === BOOK SEARCH ===
        if any(word in question_lower for word in ['search', 'find', 'look for', 'have', 'get', 'where is', 'book on', 'books about']):
            search_terms = question_lower
            for term in ['search', 'find', 'look for', 'do you have', 'where is', 'get me', 'book named', 'book called', 'books about', 'book on']:
                search_terms = search_terms.replace(term, '')
            search_terms = search_terms.strip().strip('?')
            
            if len(search_terms) < 2:
                return "🔍 **Book Search**\n\nPlease tell me what book you're looking for. Examples:\n• 'Find Python Programming'\n• 'Search for Data Science'\n• 'Do you have any books on Machine Learning?'"
            
            results = self.semantic_search(search_terms, n=5)
            if results:
                response = f"📘 **SEARCH RESULTS** - '{search_terms}'\n\n"
                response += f"Found {len(results)} relevant book(s):\n\n"
                for idx, book in enumerate(results, 1):
                    response += f"{idx}. **{book['title']}**\n"
                    response += f"   📚 Category: {book['category']}\n"
                    response += f"   🔗 Access No: {book['access_no']}\n"
                    response += f"   📊 Match Score: {book['relevance']}%\n\n"
                response += "*Results ranked by relevance to your search*"
                return response.strip()
            return f"❌ No exact matches found for '{search_terms}'.\n\nTry:\n• Using simpler keywords\n• Different book titles\n• Or ask for category recommendations"
        
        # === DEFAULT: SMART FALLBACK WITH SEMANTIC SEARCH ===
        results = self.semantic_search(question_lower, n=3)
        if results and results[0].get('relevance', 0) > 20:
            top_book = results[0]
            response = f"🔎 **RELATED BOOK FOUND:**\n\n"
            response += f"**{top_book['title']}**\n\n"
            response += f"📚 Category: {top_book['category']}\n"
            response += f"🔗 Access Number: {top_book['access_no']}\n"
            response += f"📊 Relevance: {top_book['relevance']}%\n"
            if len(results) > 1:
                response += f"\n**Related books:**\n"
                for r in results[1:]:
                    response += f"  • {r['title']} ({r['category']})\n"
            return response.strip()
        
        # Helpful fallback following FALLBACK_PROMPT
        return """**Welcome to LibBot - Your Library Assistant! 📚**

I'm here to help with library-related questions. I can assist with:

**📖 BOOK SEARCH**
  "Find [book title]" or "Search for [topic]"

**📊 LIBRARY STATISTICS**  
  "How many books do we have?"
  "Show library statistics"

**⭐ POPULAR BOOKS**
  "What are the most borrowed books?"
  "Show trending books"

**📚 CATEGORIES**
  "What categories are available?"
  "Books on [subject]"

**👥 DEPARTMENTS**
  "Department statistics"
  "Which department borrows most?"

**What would you like to know about our library?**
"""
