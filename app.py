import torch
from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os

app = Flask(
    __name__,
    template_folder="assets/gui/templates",
    static_folder="assets/gui/static"
)

# ── Load model ───────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "assets/models/saved_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print(f"Model loaded from {MODEL_PATH} on {device}")

# ── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("user_text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        pred    = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "AI-Generated" if pred == 1 else "Human"

    return jsonify({
        "label":      label,
        "confidence": round(confidence, 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)