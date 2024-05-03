import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

#Create a flask app
app = Flask(__name__)

#load pickle model
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def Home():
    return render_template("index.html")

# # Route to serve styles.css
# @app.route("/styles.css")
# def styles():
#     return app.send_static_file("styles.css")


@app.route("/predict",methods=["POST"])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)

    return render_template("index.html",prediction_text = "The net hourly electrical energy output of the plant is{}".format(prediction))

if  __name__ == "__main__" :
    app.run(debug=True)  
 