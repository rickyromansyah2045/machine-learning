import tensorflow as tf
import string 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, url_for

# Load the dataset
data = open('./search-keywords-cat.txt').read()

def clean_text(doc):
  tokens = doc.split()
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [word.lower() for word in tokens]
  return tokens

tokens = clean_text(data)

length = 5 + 1
lines = []

for i in range(length, len(tokens)):
  seq = tokens[i-length:i]
  line = ' '.join(seq)
  lines.append(line)
  if i > 200000:
    break

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

# Load the dataset
model = load_model("./text_generation_model (new).h5")

def generate_text_seq(seed_text, model, tokenizer, text_seq_length, n_words):
  text = []

  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    # y_predict = model.predict_classes(encoded)
    y_predict = np.argmax(model.predict(encoded), axis=1)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)

# if __name__ == '__main__':
#     seed_text = input("Search: ")

#     print("Recommendation: ")
#     print(generate_text_seq(seed_text, model, tokenizer, 5, 5))

# import os

# Initialize flask app
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def generate():
    if request.method == 'POST':
        search = request.form['message']
        recommendation = generate_text_seq(search, model, tokenizer, 5, 5)
    else:
        search = " "
        recommendation = " "

    return render_template('index.html',
                           seed = search,
                           generation = recommendation)

if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    seed_text = input("Search: ")

    print("Recommendation: ")
    print(generate_text_seq(seed_text, model, tokenizer, 5, 5))
    app.run(host='0.0.0.0', port=5000, debug=True)

