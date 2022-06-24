import numpy as np
from flask import Flask, request,jsonify,render_template,flash
import pickle

app = Flask(__name__)
lm = pickle.load(open('lm.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':  
        #feature = [float(i) for i in request.form.values()]
        try:
            
            weight=float(request.form['weight'])
            height=float(request.form['height'])
            neck=float(request.form['neck'])
            chest=float(request.form['chest'])
            abdomen=float(request.form['abdomen'])
            hip=float(request.form['hip'])
            thigh=float(request.form['thigh'])
            knee=float(request.form['knee'])
            ankle=float(request.form['ankle'])
            bicep=float(request.form['bicep'])
            forearm=float(request.form['forearm'])
            wrist=float(request.form['wrist'])
        
        except Exception as e:
            print('error has occured!')
            print(e)
        #feature = [np.array(feature)]
        val=np.array([weight,height,neck,chest,abdomen,hip,thigh,knee,ankle,bicep,forearm,wrist]).reshape(1,-1)
        y_pred = lm.predict(val)
        bmi=((weight/2.2)/((height*0.0254)**2))
        return render_template('index1.html', prediction_text= "bodyfat is {}".format(y_pred),prediction_bmi="bmi is {}".format(bmi))

if __name__ == "__main__":
    app.run(debug=True)