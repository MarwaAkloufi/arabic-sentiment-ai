from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Sentiment Analysis AI Microservice")

try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")

def clean_arabic_text(text: str) -> str:
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text) # إزالة التشكيل
    text = re.sub(r'ـ+', '', text)                           # إزالة التطويل
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)                      # توحيد الألف
    text = re.sub(r'ى', 'ي', text)                           # توحيد الياء
    text = re.sub(r'ؤ|ئ', 'ء', text)                         # توحيد الهمزات
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(payload: TextRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    cleaned = clean_arabic_text(payload.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    probabilities = model.predict_proba(vectorized)[0]
    classes = model.classes_.tolist()
    probs = {str(c): float(p) for c, p in zip(classes, probabilities)}
    pos_prob = probs.get('pos', 0.0)
    neg_prob = probs.get('neg', 0.0)
    confidence = float(max(probabilities))

    label = "pos" if prediction == 'pos' else "neg"
    
    return {
        "text": payload.text,
        "cleaned_text": cleaned,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "pos": round(pos_prob, 4),
            "neg": round(neg_prob, 4)
        }
    }