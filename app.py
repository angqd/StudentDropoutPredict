from flask import Flask, request, render_template, jsonify
import pandas as pd 
import numpy as np 
import pickle
#load the trained model 
model  = pickle.load(open("xgbmodel.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    
        data = request.get_json()

        # Print the received data to verify it
       # print("Received data:", data)

        # Convert the received data into a DataFrame
        x = pd.DataFrame(data, index=[0])

        # Print the DataFrame to verify its contents
        
        x = x.rename(columns={
            'application_mode': 'Application mode',
            'attendance': 'Daytime/evening attendance',
            'prev_qualification': 'Previous qualification',
            'nationality': 'Nacionality',
            'mother_qualification': "Mother's qualification",
            'father_qualification': "Father's qualification",
            'mother_occupation': "Mother's occupation",
            'father_occupation': "Father's occupation",
            'displaced': 'Displaced',
            'eductaional_needs': 'Educational special needs',
            'debtor': 'Debtor',
            'fees_uptodate': 'Tuition fees up to date',
            'gender': 'Gender',
            'scholarship': 'Scholarship holder',
            'age': 'Age at enrollment',
            'International': 'International',
            'unemployment_rate': 'Unemployment rate',
            'inflation_rate': 'Inflation rate',
            'gdp': 'GDP'
             })
        
        x = x.astype(float)
        #reordering the columns 
        colsWhen = model.get_booster().feature_names
        x = x[colsWhen]

       

        # Ensure that DataFrame x is not None
        if x is None:
            return jsonify({'error': 'DataFrame is None'}), 500

        # Assuming you have loaded your model, perform prediction
         
        print("DataFrame x:", x)
        prediction = model.predict(x)
        print(prediction[0])
         # Sending the prediction result back to the frontend
        result = 'Graduate' if prediction[0] == 1 else 'Dropout'
        return jsonify({'prediction_text': result})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')