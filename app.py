from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


training_texts = [
    # Happy
    "I am very happy today",
    "I feel wonderful and excited",
    "This is a fantastic day",
    "I love this amazing experience",
    "I am joyful and pleased",

    # Sad
    "I am sad and disappointed",
    "I feel lonely and unhappy",
    "This news made me cry",
    "I am feeling very sad today",
    "I miss my friends and feel lonely",

    # Angry
    "I am angry about this situation",
    "This makes me furious",
    "I am very annoyed",
    "I hate what happened",
    "I feel angry and frustrated",

    # Fear
    "I am scared about the future",
    "I am afraid of this situation",
    "I feel nervous and frightened",
    "This is terrifying",
    "I am worried and scared",

    # Surprise
    "I am surprised by the result",
    "Wow I did not expect this",
    "This is an unexpected surprise",
    "I cannot believe what happened",
    "I am shocked by the news",

    # Neutral
    "The weather is normal today",
    "I am going to college",
    "The meeting starts at ten",
    "I have a class tomorrow",
    "The book is on the table"
]

training_labels = [
    "Happy", "Happy", "Happy", "Happy", "Happy",
    "Sad", "Sad", "Sad", "Sad", "Sad",
    "Angry", "Angry", "Angry", "Angry", "Angry",
    "Fear", "Fear", "Fear", "Fear", "Fear",
    "Surprise", "Surprise", "Surprise", "Surprise", "Surprise",
    "Neutral", "Neutral", "Neutral", "Neutral", "Neutral"
]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_texts)

model = MultinomialNB()
model.fit(X, training_labels)


emotion_descriptions = {
    "Happy": "The text expresses happiness or positive feelings.",
    "Sad": "The text expresses sadness, loneliness, or disappointment.",
    "Angry": "The text expresses anger, frustration, or annoyance.",
    "Fear": "The text expresses fear, worry, or nervousness.",
    "Surprise": "The text expresses surprise or unexpected feelings.",
    "Neutral": "The text expresses a neutral or ordinary statement."
}


def detect_emotion(text):
    transformed_text = vectorizer.transform([text])

    prediction = model.predict(transformed_text)[0]

    probabilities = model.predict_proba(transformed_text)[0]
    confidence = round(max(probabilities) * 100, 2)

    description = emotion_descriptions[prediction]

    return prediction, confidence, description


@app.route("/", methods=["GET", "POST"])
def index():

    text = ""
    emotion = None
    confidence = None
    description = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if text:
            emotion, confidence, description = detect_emotion(text)

    return render_template(
        "index.html",
        text=text,
        emotion=emotion,
        confidence=confidence,
        description=description
    )


if __name__ == "__main__":
    app.run(debug=True)
