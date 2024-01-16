from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


app = Flask(__name__)

data = pd.read_excel('dataset.xlsx')

# Chia thành dữ liệu huấn luyện và dữ liệu kiểm thử
X_train, X_test, y_train, y_test = train_test_split(data['Title'], data['Category'], test_size=0.2, random_state=42)

# Chuyển đổi văn bản thành ma trận đặc trưng sử dụng Bag of Words
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Sử dụng mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text_to_predict = data['text']
    text_to_predict_vec = vectorizer.transform([text_to_predict])
    prediction = model.predict(text_to_predict_vec)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
