from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the lookup_tables
with open('lookup_tables.pkl', 'rb') as f:
    lookup_tables = pickle.load(f)

# Load the best model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Feature engineering function
def feature_engineering(df, is_train=True):
    # Convert data types
    num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
    if is_train:
        cat_columns.append('Loan_Status')
    
    df[num_columns] = df[num_columns].astype(float)
    df[cat_columns] = df[cat_columns].astype('category')
    
    df['LoanAmount'] = df['LoanAmount'] * 1000
    df['LoanAmount'] = np.log(df['LoanAmount'])
    
    if is_train:
        df['Loan_Status_number'] = df['Loan_Status'].map({'Y': 1, 'N': 0}).astype(int)

    temp_married_series = df['Married'].map({'Yes': 2, 'No': 1}).astype('Int64')
    df['Married_number'] = temp_married_series.fillna(1).astype(int)
    
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome'] = np.log(df['TotalIncome'])    
    temp_dependents_series = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3}).astype('Int64')
    df['Dependents_number'] = temp_dependents_series.fillna(0).astype(int)
    df['family_size'] = df['Dependents_number'] + df['Married_number']
    
    df['income_per_person'] = df['TotalIncome'] / df['family_size']
    df['income_share'] = df['CoapplicantIncome'] / df['ApplicantIncome']
    df['LoanAmount_monthly'] = df['LoanAmount'] / df['Loan_Amount_Term'].astype(float) 
    df['LoanAmount_monthly_to_income'] = df['LoanAmount_monthly'] / df['TotalIncome']
    df['LoanAmount_monthly_per_person'] = df['LoanAmount_monthly'] / df['family_size']

    if is_train:
        global lookup_tables
        lookup_tables = {}
        for column in cat_columns:
            if column != 'Loan_Status':
                lookup_tables[column] = df.groupby(column)['Loan_Status_number'].agg(['mean', 'count']).reset_index()
                lookup_tables[column].columns = [column, f'{column}_mean', f'{column}_count']
    
    for column in cat_columns:
        if column != 'Loan_Status':
            df = df.merge(lookup_tables[column], on=column, how='left')
            df = df.drop(column, axis=1)

    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)


    if is_train:
        df.drop('Loan_Status_number', axis=1, inplace=True)
    
    df.drop('Dependents_number', axis=1, inplace=True)
    df.drop('Married_number', axis=1, inplace=True)
    
    if is_train:
        return df, lookup_tables
    else:
        return df

    

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])

    # Pre-process the data using the feature engineering function
    df_processed = feature_engineering(df, is_train=False)
    
    # Make predictions
    prediction = best_model.predict(df_processed)

    # Convert the prediction to 'N' or 'Y'
    result = 'Y' if prediction[0] == 1 else 'N'

    # Return the result as JSON
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
