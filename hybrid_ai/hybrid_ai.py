import os
import sqlite3
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pyttsx3
from duckduckgo_search import DDGS

class HybridAI:
    def forget_pdf(self, pdf_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Get the filename for the given pdf_id
        c.execute("SELECT filename FROM pdf_files WHERE id=?", (pdf_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return f"No PDF found with ID {pdf_id}"
        filename = row[0]
        # Delete all memory entries related to this PDF
        c.execute("DELETE FROM memory WHERE source=?", (f"pdf:{filename}",))
        # Delete the PDF record
        c.execute("DELETE FROM pdf_files WHERE id=?", (pdf_id,))
        conn.commit()
        conn.close()
        return f"Forgot PDF '{filename}' and all related memory."
    def __init__(self, db_path="hybrid_ai_v2.db"):  # Fixed constructor name
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
                        # split into smaller chunks
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

if __name__ == "__main__":
    ai = HybridAI()
    print("Hybrid AI v2 (PDF chunking + summarization).")
    print("Commands: ingest_pdf <path> | say | ask | list | list_pdfs | forget <id> | forget_pdf <pdf_id> | tts_on | tts_off | exit")

    while True:
        cmd = input("\n> ").strip()
        if cmd.startswith("ingest_pdf"):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 2:
                path = parts[1]
                print(ai.ingest_pdf(path))
            else:
                print("Usage: ingest_pdf <path>")
        elif cmd == "say":
            text = input("Enter text: ")
            print(ai.say(text))
        elif cmd == "ask":
            question = input("Question: ")
            print(ai.ask(question))
        elif cmd == "list":
            for mem in ai.list_memory():
                print(mem)
        elif cmd == "list_pdfs":
            for pdf in ai.list_pdfs():
                print(pdf)
        elif cmd.startswith("forget"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                print(ai.forget(int(parts[1])))
            else:
                print("Usage: forget <memory_id>")
        elif cmd.startswith("forget_pdf"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                print(ai.forget_pdf(int(parts[1])))
            else:
                print("Usage: forget_pdf <pdf_id>")
        elif cmd == "tts_on":
            print(ai.toggle_tts(True))
        elif cmd == "tts_off":
            print(ai.toggle_tts(False))
        elif cmd == "exit":
            break
        else:
            print("Unknown command.")
