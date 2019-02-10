from flask import Flask, render_template, request
from keluhan import preprocess_twitter, cek_keluhan

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        text = request.form['keluhan']
        vec_option = request.form['vec_option']
        keluhan = []
        keluhan.append(text)

        arr = ['Doc-term', 'TF IDF', 'TF IDF normalized']
        keluhan.append(arr[int(vec_option) - 1])
        
        tweet = preprocess_twitter(text)
        for i in range(3):
            is_keluhan = cek_keluhan(tweet, vec_option, i + 1)
            keluhan.append('ya' if is_keluhan else 'bukan')

        keluhans = []
        keluhans.append(keluhan)
        return render_template('index.html', keluhans = keluhans)
    return render_template('index.html')
