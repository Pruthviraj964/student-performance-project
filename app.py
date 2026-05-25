from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = [float(x) for x in request.form.values()]

        prediction = model.predict([features])

        output = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Math Score: {output}'
        )

    except:
        return render_template(
            'index.html',
            prediction_text='Error in prediction'
        )

if __name__ == "__main__":
    app.run(debug=True)