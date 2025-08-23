from flask import Flask, request
import hybrid_ai   # load your AI code

app = Flask(_name_)

@app.route("/")
def home():
    return "Hello! Visit /ask?question=hello to talk to Hybrid AI."

@app.route("/ask")
def ask():
    question = request.args.get("question", "")
    answer = hybrid_ai.run_ai(question)  # use your AI function
    return f"Hybrid AI says: {answer}"

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)