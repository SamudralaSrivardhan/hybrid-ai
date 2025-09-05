import os
import sqlite3
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pyttsx3
from duckduckgo_search import DDGS
from flask import Flask, request, jsonify

app = Flask(__name__)

class HybridAI:
    def __init__(self, db_path="hybrid_ai_v2.db"):
        self.db_path = db_path
        self._init_db()
        self.vectorizer = TfidfVectorizer()
        self.tts_enabled = False
        self.engine = pyttsx3.init()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS memory
                     (id INTEGER PRIMARY KEY, source TEXT, content TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS pdf_files
                     (id INTEGER PRIMARY KEY, filename TEXT)''')
        conn.commit()
        conn.close()

    def _save_memory(self, source, content):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO memory (source, content) VALUES (?, ?)", (source, content))
        conn.commit()
        conn.close()

    def _load_all_memory(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT content FROM memory")
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]

    def _search_local_memory(self, query):
        memories = self._load_all_memory()
        if not memories:
            return None
        tfidf = self.vectorizer.fit_transform(memories + [query])
        cosine_sim = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        top_index = cosine_sim.argmax()
        if cosine_sim[top_index] > 0.2:
            return memories[top_index]
        return None

    def ingest_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            return "PDF file not found."
        text_chunks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        for chunk in text.split("\n"):
                            if chunk.strip():
                                text_chunks.append(chunk.strip())
        except Exception as e:
            return f"Error reading PDF: {e}"
        for chunk in text_chunks:
            self._save_memory(f"pdf:{os.path.basename(pdf_path)}", chunk)
        self._save_pdf_record(pdf_path)
        return f"Ingested {len(text_chunks)} text chunks from {pdf_path}"

    def _save_pdf_record(self, pdf_path):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO pdf_files (filename) VALUES (?)", (os.path.basename(pdf_path),))
        conn.commit()
        conn.close()

    def search_online(self, query):
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append(r.get("body") or r.get("title", ""))
            return " ".join(results) if results else None
        except Exception:
            return None

    def ask(self, query):
        local_answer = self._search_local_memory(query)
        if local_answer:
            self._speak(local_answer)
            return local_answer
        online_answer = self.search_online(query)
        if online_answer:
            self._save_memory("online", online_answer)
            self._speak(online_answer)
            return online_answer
        return "Sorry, I couldn’t find an answer."

    def say(self, text):
        self._save_memory("user", text)
        return "Noted and stored in memory."

    def list_memory(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, source, content FROM memory")
        rows = c.fetchall()
        conn.close()
        return rows

    def forget(self, memory_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM memory WHERE id=?", (memory_id,))
        conn.commit()
        conn.close()
        return f"Deleted memory with ID {memory_id}"

    def forget_pdf(self, pdf_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT filename FROM pdf_files WHERE id=?", (pdf_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return f"No PDF found with ID {pdf_id}"
        filename = row[0]
        c.execute("DELETE FROM memory WHERE source=?", (f"pdf:{filename}",))
        c.execute("DELETE FROM pdf_files WHERE id=?", (pdf_id,))
        conn.commit()
        conn.close()
        return f"Forgot PDF '{filename}' and all related memory."

    def list_pdfs(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, filename FROM pdf_files")
        rows = c.fetchall()
        conn.close()
        return rows

    def toggle_tts(self, enable):
        self.tts_enabled = enable
        return f"TTS {'enabled' if enable else 'disabled'}."

    def _speak(self, text):
        if self.tts_enabled:
            self.engine.say(text)
            self.engine.runAndWait()

ai = HybridAI()

@app.route('/')
def index():
    return "Hybrid AI v2 API is running!"

@app.route('/ingest_pdf', methods=['POST'])
def api_ingest_pdf():
    data = request.json
    pdf_path = data.get('pdf_path')
    result = ai.ingest_pdf(pdf_path)
    return jsonify({'result': result})

@app.route('/ask', methods=['POST'])
def api_ask():
    data = request.json
    query = data.get('query')
    result = ai.ask(query)
    return jsonify({'result': result})

@app.route('/say', methods=['POST'])
def api_say():
    data = request.json
    text = data.get('text')
    result = ai.say(text)
    return jsonify({'result': result})

@app.route('/list_memory', methods=['GET'])
def api_list_memory():
    result = ai.list_memory()
    return jsonify({'memory': result})

@app.route('/list_pdfs', methods=['GET'])
def api_list_pdfs():
    result = ai.list_pdfs()
    return jsonify({'pdfs': result})

@app.route('/forget', methods=['POST'])
def api_forget():
    data = request.json
    memory_id = data.get('memory_id')
    result = ai.forget(memory_id)
    return jsonify({'result': result})

@app.route('/forget_pdf', methods=['POST'])
def api_forget_pdf():
    data = request.json
    pdf_id = data.get('pdf_id')
    result = ai.forget_pdf(pdf_id)
    return jsonify({'result': result})

@app.route('/toggle_tts', methods=['POST'])
def api_toggle_tts():
    data = request.json
    enable = data.get('enable', True)
    result = ai.toggle_tts(enable)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
