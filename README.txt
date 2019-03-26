A simple program to predict if a transaction is fraudulent or not. 

Run ./download_data.sh
python src/model.py ~/Coding/Portfolio/Fraudulent_Transactions/mlp/data/creditcard.csv ~/Coding/Portfolio/Fraudulent_Transactions/mlp/output/predictions.csv ~/Coding/Portfolio/Fraudulent_Transactions/mlp/src/fraud_model.pkl


where args are input_data, predictions, model  
