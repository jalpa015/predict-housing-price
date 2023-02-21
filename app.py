from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/hello/<name>')
def hello_name(name):
    return 'Hello %s!' % name


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # get parameters from request
    # make prediction
    # return prediction
    print("IN Predict")
    try:
        params = request.form.to_dict()
        print(params)
        for key in params:
            params[key] = float(params[key])
        print(params)
        model = joblib.load('model/model.pkl')
        output = model.predict([list(params.values())])
        print("Output ", output)
        return render_template('index.html', prediction="{:.2f}".format(round(output[0], 2)))
    except:
        return 'Error'


if __name__ == "__main__":
    app.run(port=5000, debug=True)
