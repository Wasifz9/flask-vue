import pickle 
from flask import Flask, jsonify, request 
from flask_cors import CORS, cross_origin 
feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('wine_model.pickle', 'rb'))




@app.route("/")
@cross_origin()
def helloworld():
        return "Hellooooo Worlddddd!"


@app.route("/wine", methods = ["GET"])

def predWine():
    if request.method == 'GET':
        # features
        a = float(request.args.get('alcohol'))
        b = float(request.args.get('malic_acid'))
        c = float(request.args.get('ash'))
        d = float(request.args.get('alcalinity_of_ash'))
        e = float(request.args.get('magnesium'))
        f = float(request.args.get('total_phenols'))
        g = float(request.args.get('flavanoids'))
        h = float(request.args.get('nonflavanoid_phenols'))
        i = float(request.args.get('proanthocyanins'))
        i2= float(request.args.get('proanthocyanins2'))
        j = float(request.args.get('color_intensity'))
        k = float(request.args.get('hue'))
        l = float(request.args.get('od315_of_diluted_wines'))
        m = float(request.args.get('proline'))

        final_features = ([[a, b, c, d, e, f, g, h, i, j, k, l, m]])

        # model predicition
        prediction = model.predict(final_features)

    return jsonify(str("Class  " + str(prediction[0])))


if __name__ == '__main__':
        app.run(debug=True)
