# Text Emotion Detection System

A web-based Natural Language Processing (NLP) application that analyzes text and detects the emotion expressed in it.

The system uses **TF-IDF Vectorization** and a **Multinomial Naive Bayes** machine learning model to classify text into different emotion categories.

## Features

* Detects emotions from text
* Supports six emotion categories
* Displays prediction confidence
* Simple and user-friendly web interface
* Uses machine learning for text classification
* Responsive design for different screen sizes

## Emotion Categories

The system detects the following emotions:

* Happy
* Sad
* Angry
* Fear
* Surprise
* Neutral

## Technologies Used

* Python
* Flask
* Scikit-learn
* HTML
* CSS
* TF-IDF
* Multinomial Naive Bayes

## Project Structure

```text
Text-Emotion-Detection-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## Installation

### 1. Clone or download the project

Open the project folder in a terminal.

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Open the application

Open the following address in a web browser:

```text
http://127.0.0.1:5000
```

## How It Works

The application follows these steps:

1. The user enters a sentence or paragraph.
2. The Flask application receives the text.
3. The text is converted into numerical features using TF-IDF.
4. The trained Naive Bayes model analyzes the features.
5. The model predicts the most likely emotion.
6. The application displays the detected emotion and confidence percentage.

## Example

### Input

```text
I am very happy because I got good marks.
```

### Output

```text
Detected Emotion: Happy
```

The application also displays a confidence percentage for the prediction.

## Applications

Text emotion detection can be used in:

* Social media analysis
* Customer feedback analysis
* Chatbot systems
* Online review analysis
* User experience analysis
* Educational applications
* Opinion mining

## Advantages

* Simple web interface
* Fast prediction
* Easy to understand
* Uses machine learning
* Can be extended with larger datasets

## Limitations

This project uses a small demonstration dataset. Therefore, it is intended for educational purposes and may not provide highly accurate predictions for all types of real-world text.

For a production-level system, the model can be trained using a much larger and more diverse emotion dataset.

## Purpose

This project demonstrates the use of **Natural Language Processing and Machine Learning for emotion classification** through a web application.

## Author

B.E. CSE Student
