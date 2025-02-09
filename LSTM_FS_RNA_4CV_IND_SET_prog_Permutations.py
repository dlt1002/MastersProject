#code runs LSTM with 3 layers which can have defined number of nodes in each Layer
#2 * LSTM Layer
#1 * Dense Layer
#Continuous output to map to Updrs Values, MSE Loss function, MAE for accuracy
#Loads all RNA features but can implement selection via a file from MRMR selection either scaleds or dsictetzed (Binned)

#Splits data into 5 folds 4 used for 4 blocks of sucessive training without re-initialisation, 5th used as holdout for testing


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization, Activation, Masking
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from itertools import permutations
from numpy import array
import matplotlib.pyplot as plt


#function to load the data from individiaul files
def load_data (data_dir):
    load_df = pd.DataFrame()
    patient_labels = []
    sample_ct = 0
    for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_dir, filename))
                #get patient labels and append
                patient_labels.append(df['updrs_totscore'].tolist())
                #print(patient_labels)
                #drop data info columns
                df.drop(df.columns[[0,1,2,3,4,5]], axis = 1,inplace=True)
                visits = df.shape[0]
                
                #increase sample ct
                sample_ct +=1

                load_df = pd.concat([load_df,df])

    #get feature size
    features = load_df.shape[1]

    # Flatten the list
    patient_labels = [item for sublist in patient_labels for item in sublist]
    # add back patient labels
    load_df.insert(0, 'updrs_totscore', patient_labels)

    return load_df, visits, features, sample_ct

# function to select features based on loaded feature list
def select_features (start_df, feature_nom):
    updrs_df = start_df['updrs_totscore']
    start_df = start_df[feature_nom]
    feature_ct = len(feature_nom)
    start_df = pd.concat([updrs_df, start_df], axis = 1)
    #start_df.to_csv('start_df_post.csv', index=False)
    return start_df, feature_ct

#function to shape the data after loading
def shape_data(input_df, visits_st, sample_ct):
    sequences_temp = []
    patient_labels = []
    labels_temp = []
    for i in range(0,len(input_df), visits_st):
        # Select a range of rows based on visits
        df = input_df.iloc[i: i+ visits_st]
        #get patient label
        patient_labels = df['updrs_totscore'].tolist()
        patient_labels = patient_labels[-1]
        #drop last row of data from data
        df = df.iloc[:-1 , :]
        #redefine feature list
        visits_act = df.shape[0]
        features_act = df.shape[1]
        #print(features_act)

        #scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values)
        sequences_temp.append(scaled_data)
        labels_temp.append(patient_labels)

    #shape sequence array for LSTM (samples, timeseries, fetures) format
    sequences_temp = array(sequences_temp).reshape(sample_ct, visits_act, features_act)
    labels_temp = array(labels_temp)
        
    return sequences_temp, labels_temp, visits_act

#Function to define LSTM model
def model_define_LSTM(input_shape, model_structure):
    #input_shape=(numof_outputs, numof_features)
# define model where LSTM is also output layer
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(model_structure[0],dropout=0.2,return_sequences=True))
    model.add(LSTM(model_structure[1],dropout=0.2))
    model.add(Dense(model_structure[2],activation='relu'))
    model.add(Dense(1,activation='linear'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return(model)

#Function to output LSTM model
def output_results(outlist, outfile):
    out_df = pd.read_csv(outfile)
    new_row_df = pd.DataFrame([outlist], columns=out_df.columns)
    out_df = pd.concat([out_df, new_row_df], ignore_index=True)
    out_df.to_csv(outfile, index=False)

def output_predictions(y_tr, y_tr_pred, y_val, y_val_pred, fil_nom_val, fil_nom_tr, comments):
    
    #Combine the lists into a DataFrame
    df_tr = pd.DataFrame({'y_train': y_tr,'y_train_predicted': y_tr_pred})
    #df_tr.to_csv('predictions_tr.csv', index=False)
    df_val = pd.DataFrame({'y_validation': y_val, 'y_validation_predicited': y_val_pred})
    #df_val.to_csv(fil_nom, index=False)


    # Export DataFrame with comments as header lines
    #validation
    with open(fil_nom_val, "w") as f:
        # Write the comments
        for comment in comments:
            f.write(comment + "\n")
        # Save the DataFrame to the file, appending it after the comments
        df_val.to_csv(f, index=False)

    # Export DataFrame with comments as header lines
    #training
    with open(fil_nom_tr, "w") as f:
        # Write the comments
        for comment in comments:
            f.write(comment + "\n")
        # Save the DataFrame to the file, appending it after the comments
        df_tr.to_csv(f, index=False)


    return



#main
def main():

    #Steps to implement
    #1) Load the data
    #data_dir = '../data/Pat_Data_Ind_4'
    data_dir = 'Pat_Data_Ind_4'
    fullfeature_df, visits_total, features_total, sample_ct = load_data (data_dir)
    print(features_total)
    print(visits_total)
    print(sample_ct)
    
    #feature selection disabled
    #1b) Select the features
    fs_file = 'MRMR_FS_20_5_Disc_100.csv'
    fs_df = pd.read_csv(fs_file)
    #print(fs_df)
    
    fs_list = fs_df['feature_names'].to_list()
    #print(fs_list)

    fullfeature_df, features_total = select_features(fullfeature_df, fs_list)
    
    print(features_total)

    #fs_file = 'FullFeatures.csv'

    #create file for output with custom name
    #fil_out = '_'.join(fs_file) + '_'+ str(i) + '_output.csv'
    fil_out = os.path.splitext(fs_file)[0] + '_LSTM_'+ str(visits_total) + 'V_4CV_4V_IND_prog_output.csv'

    outtodf = ['perm', 'holdout_index', 'loss_tr', 'mae_tr', 'loss_val', 'mae_val', 'loss_ind', 'mae_ind']
    df = pd.DataFrame(columns=outtodf)
    df.to_csv(fil_out, index=False)

    
    
    #2) shape the data
    sequences, labels, visits = shape_data(fullfeature_df, visits_total, sample_ct)

    X = sequences
    y = labels
    
    #4) create the model
    # Define input shape
    input_shape = (visits, features_total + 1)

    #cycle list of variables
    #db_list = [8,16,32,64,128,256]
    #lstm_lst = [8,16,32,64,128,256]
    #dense_units = [8,16,32,64,128,256]
    
    #testing cycle list
    db_list = [32]
    lstm_lst = [32]
    dense_units = [32]


    for db in db_list:
        for lst in lstm_lst:
            for dus in dense_units:
                # Build the LSTM model
                #5) Train the model
                # Train the model
                #intialise the model prior to performing the fold split on the data

                #Split off 20% of the data as a true independent set X_ind & y_ind not involved in the training
                X_train, X_ind, y_train, y_ind = train_test_split(X, y, test_size=0.2, random_state=42)
 
                
                #set up the data into 5 folds
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                # Split the data into 5 folds the 80% of data used for training
                folds = []
                for train_index, test_index in kf.split(X_train):
                    folds.append((X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index]))

                # Perform sequential training using all permutations of the folds
                fold_indices = list(range(5))
                permutations_of_folds = list(permutations(fold_indices, 5))

                results = []  # To store results of each permutation
                
                perm_ct = 1
                for perm in permutations_of_folds:
                    print(f"Processing permutation: {perm}")
                    print(f"Permutation No: {perm_ct}")
    
                    # Get the holdout set
                    holdout_index = perm[-1]
                    X_holdout, y_holdout = folds[holdout_index][2], folds[holdout_index][3]
    
                    # Use the other folds in the order of the permutation for training
                    model = model_define_LSTM(input_shape, [lst, dus, db])
                    # Compile the model for regression (continuous output)
                    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

                    for train_index in perm[:-1]:
                        X_train, y_train = folds[train_index][0], folds[train_index][1]
                        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)  # Train on this fold sequentially

                    # Evaluate on the last training set
                    loss_tr, mae_tr = model.evaluate(X_train, y_train, verbose=0)
                    print(f"Last Training Fold: Loss: {loss_tr}, MAE: {mae_tr}")
                    
                    # Evaluate on the holdout set
                    loss_val, mae_val = model.evaluate(X_holdout, y_holdout, verbose=0)
                    print(f"Holdout Fold: {holdout_index}, Loss: {loss_val}, MAE: {mae_val}")
                    
                    # Evaluate on the independent set
                    loss, mae = model.evaluate(X_ind, y_ind, verbose=0)
                    print(f"Ind Fold: Loss: {loss}, MAE: {mae}")
                    results.append((perm, holdout_index, loss_tr, mae_tr, loss_val, mae_val, loss, mae))
                    perm_ct +=1
                    
                    out_data = [perm, holdout_index, loss_tr, mae_tr, loss_val, mae_val, loss, mae]
                    output_results(out_data, fil_out)

    # Print final results
    #print("\nFinal Results:")
    #for perm, holdout_index, loss_tr, mae_tr, loss, mae in results:
    #    print(f"Permutation: {perm}, Holdout Fold: {holdout_index}, Loss: {loss:.4f}, MAE: {mae:.4f}")
# Ensure this block only runs when the script is executed directly
if __name__ == "__main__":
    main()
