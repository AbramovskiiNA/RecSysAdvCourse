import json

from flask import Flask, request

from model import ModelServing

app = Flask(__name__)
model_serving = ModelServing()


@app.route('/users')
def get_known_users():
    return json.dumps(model_serving.users)


@app.route('/predict')
def predict():
    return json.dumps(model_serving.predict(
        int(request.args['user_id']),
        int(request.args['k'])
    ))


@app.route('/shutdown', methods=['GET', 'POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run(host="0.0.0.0")
