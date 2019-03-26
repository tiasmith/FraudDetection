from __future__ import print_function
import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import pickle
import sklearn
import sys


class FraudModel:

    def __init__(self, model_location):
        with open(model_location, 'rb') as f:
            self.model = pickle.load(f)

    def predict_proba(self, X_new, augment=True):
        if augment:
            X_new = self.engineer_features(X_new)

        return X_new, self.model.predict_proba(X_new)

    def engineer_features(self, df):
        df = df.drop('Class', axis=1)
        df['Duplicated'] = df.duplicated().astype(int) 
        df['Amt_Below_2500'] = (df.Amount < 2500).astype(int)
        df['V4_Above_5'] = (df.V4 > 5).astype(int)
        df['V9_Below_5'] = (df.V5 < 5).astype(int)
        df['V10_Below_11'] = (df.V10 < 11).astype(int)
        df['V11_Above_4'] = (df.V11 > 4).astype(int)
        df['V12_Below_neg3'] = (df.V12 < -3).astype(int)
        df['V14_Below_neg5'] = (df.V14 > -5).astype(int)
        df['V16_Below_neg4'] = (df.V16 < -4).astype(int)
        df['V17_Below_neg3'] = (df.V17 < -3).astype(int)
        df['V18_Below_4'] = (df.V18 < -4).astype(int)
        return df


def main(data_location, output_location, model_location, augment=True):
    # Read dataset
    df = pd.read_csv(data_location)

    # Initialize model
    fraud_model = FraudModel(model_location)

    # Make prediction
    df, pred = fraud_model.predict_proba(df)
    pred = [p[1] for p in pred]

    # Add prediction to dataset
    df['prediction'] = pred

    # Save dataset after making predictions
    df.to_csv(output_location, index=None)



if __name__ == '__main__':
    main( *sys.argv[1:] )
