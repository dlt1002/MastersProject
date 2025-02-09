#code runs LSTM with 3 layers which can have defined number of nodes in each Layer
#2 * LSTM Layer
#1 * Dense Layer
#Continuous output to map to Updrs Values, MSE Loss function, MAE for accuracy
#Loads all RNA features but can implement selection via a file from MRMR selection either scaleds or dsictetzed (Binned)
#Loads all imaging features and combines then with RNA data set availiable for equivalent RNA visit features
#Standard 5 fold cross validation used for the data with re-initialisation after each folds training

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
from numpy import array
import matplotlib.pyplot as plt


#function to load the data from individiaul files
def load_data_rna (data_dir, pat_codelst):
    load_df = pd.DataFrame()
    patient_labels = []
    sample_ct = 0
    for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_dir, filename))
                #check if the patno is in the patno list i.e. has imaging data
                if any(df['PATNO'].isin(pat_codelst)):
                    #get patient labels and append
                    patient_labels.append(df['updrs_totscore'].tolist())
                    #print(patient_labels)
                    #drop data info columns
                    df.drop(df.columns[[1,3,4,5]], axis = 1,inplace=True)
                    visits = df.shape[0]
                
                    #increase sample ct
                    sample_ct +=1

                    load_df = pd.concat([load_df,df])

    #get feature size
    features = load_df.shape[1]

    # Flatten the list
    patient_labels = [item for sublist in patient_labels for item in sublist]
    # add back patient labels
    load_df.insert(1, 'updrs_totscore', patient_labels)

    return load_df, visits, features, sample_ct


# function to select features based on loaded feature list
def select_features (start_df, feature_nom):
    updrs_df = start_df['updrs_totscore']
    pat_df = start_df['PATNO']
    event_df = start_df['EVENT_ID']
    start_df = start_df[feature_nom]
    feature_ct = len(feature_nom)
    start_df = pd.concat([updrs_df, start_df], axis = 1)
    start_df = pd.concat([event_df, start_df], axis = 1)
    start_df = pd.concat([pat_df, start_df], axis = 1)

    #start_df.to_csv('start_df_post.csv', index=False)
    return start_df, feature_ct

#function to load the data
def load_data_img (data_dir):
    sample_ct = 0
    df = pd.read_csv(data_dir)
    #get patient labels and append
    #patient_labels.append(df['updrs_totscore'].tolist())
    #print(patient_labels)
    #drop data info columns
    
    patient_labels = df['updrs_totscore'].to_list()
    visits = len(df['EVENT_ID'].unique().tolist())
    sample_ct = len(df['PATNO'].unique().tolist())
    pat_lst = df['PATNO'].unique().tolist()
    
    df.drop(df.columns[[1,3,4,5]], axis = 1,inplace=True)
    #visits = df.shape[0]
    #get feature size
    features = df.shape[1]

    # Flatten the list
    #patient_labels = [item for sublist in patient_labels for item in sublist]
    # add back patient labels
        # add back patient labels
    #don't reinsert updrs
    df.insert(2, 'updrs_totscore', patient_labels)
    return df, visits, features, sample_ct, pat_lst


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

#function to output results for a particular set of layer sizes
def output_results(outlist, outfile):
    out_df = pd.read_csv(outfile)
    new_row_df = pd.DataFrame([outlist], columns=out_df.columns)
    out_df = pd.concat([out_df, new_row_df], ignore_index=True)
    out_df.to_csv(outfile, index=False)


#function to output a set of predicitons for a particualr run to enable future plotting
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
    
        #Steps to implement
    #1) Load the imaging data and obtain a list of patients
    data_fil = 'Imaging3v2.csv'
    img_df, visits_total, features_total, sample_ct, patlst = load_data_img (data_fil)
    print ("IMG_Stats:")
    print(features_total)
    print(visits_total)
    print(sample_ct)

    
    #1) Load the rna data but only keep that data for patients who have imaging data
    data_dir = '../data/Pat_Data_Ind_3'
    fullfeature_df, visits_total, features_total, sample_ct = load_data_rna (data_dir, patlst)
    print ("RNA_Stats:")
    print(features_total)
    print(visits_total)
    print(sample_ct)
    
    #feature selection disabled
    #1b) Select the features
    fs_file = 'MRMR_FS_20_5_Disc_10.csv'
    fs_df = pd.read_csv(fs_file)
    #print(fs_df)
    
    fs_list = fs_df['feature_names'].to_list()
    #print(fs_list)

    #need to add 'PATNO' & 'EVENT_ID' to this list so it is reatined maybe?
    fullfeature_df, features_total = select_features(fullfeature_df, fs_list)
    
    print("RNA Features_Post_Selection")
    print(features_total)

    #need to combine RNA and imgaing features at this point
    
    fullfeature_df = pd.merge(fullfeature_df, img_df, on=['PATNO', 'EVENT_ID', 'updrs_totscore'], how='inner')
    
    

    print(img_df)
    print(fullfeature_df)
    print(fullfeature_df.columns.tolist())

    #drop patno and event id from df

    fullfeature_df.drop(fullfeature_df.columns[[0,1]], axis = 1,inplace=True)
    #drops the 10 RNA Features so we only have imaging features
    fullfeature_df.drop(fullfeature_df.columns[[1,2,3,4,5,6,7,8,9,10]], axis = 1,inplace=True)


    #img_df.drop(img_df.columns[[0,1]], axis = 1,inplace=True)
    #features_total = img_df.shape[1] - 1
    
    print(fullfeature_df)
    print(fullfeature_df.columns.tolist())
    
    #get final feature count for input
    features_total = (fullfeature_df.shape[1]) - 1


    #create file for output with custom name
    #fil_out = '_'.join(fs_file) + '_'+ str(i) + '_output.csv'
    fil_out = os.path.splitext(fs_file)[0] + '_LSTM_'+ str(visits_total) + 'V_5CV_IMG_ONLY_output.csv'

    outtodf = ['train_MSE', 'train_MAE', 'test_MSE', 'test_MAE', 'L1', 'L2', 'L3']
    df = pd.DataFrame(columns=outtodf)
    df.to_csv(fil_out, index=False)

    
    
    #2) shape the data
    sequences, labels, visits = shape_data(fullfeature_df, visits_total, sample_ct)

    #3)set up training and validation sets
    #X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)
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


    
    
    #setupbest MAE, if MAE is better than this it will output predicitions to a file for future plotting
    best_mae = 100.00

    
    #number of iterations
    n_iter = 100

    for i in range(n_iter):#do a certain number of iterations for averaging purposes
        #loops for 3 lists of sizes in a;yers
        for db in db_list:
            for lst in lstm_lst:
                for dus in dense_units:
                    # Build the LSTM model
                    #5) Train the model
                    # Train the model
                    #Set up five fold splits
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    fold = 1
                    train_mae = []  # To store metric scores from each fold
                    train_loss = []  # To store metric scores from each fold
                    val_mae = []  # To store metric scores from each fold
                    val_loss = []  # To store metric scores from each fold
    
                    for train_index, val_index in kf.split(X):
                        print(f"Training on fold {fold}...")
        
                        # Split data into training and validation sets
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
     
                        #build model
                        model = model_define_LSTM(input_shape, [lst, dus, db])
                        # Compile the model for regression (continuous output)
                        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
     
     
        
                        # Create and train the model, model is set up and trained in side the 5 fold loop so will be reinitialised each time
                        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose = False)
                        #6) output the data from the model
                        # Evaluate the model on test data & caputre output
                        temp1, temp2 = model.evaluate(X_val, y_val)
                        val_loss.append(temp1)
                        val_mae.append(temp2)
    
                        # Evaluate the model on training data & capture output
                        temp3, temp4 = model.evaluate(X_train, y_train)
                        train_loss.append(temp3)
                        train_mae.append(temp4)
    
                        #Code to output best predicitions if validation MAE is the best
                        if temp2 < best_mae:
                            comment = [f"# train_mse:{temp3} train_mae:{temp4} test_mse:{temp1} tst_mae:{temp2} L1:{db} L2:{lst} L3:{dus}"]
                            
                            fil_predict_val = os.path.splitext(fs_file)[0] + '_LSTM_val_'+ str(visits_total) + 'V_5CV__IMG_ONLY_BestMAE.csv'
                            fil_predict_tr = os.path.splitext(fs_file)[0] + '_LSTM_tr_'+ str(visits_total) + 'V_5CV__IMG_ONLY_BestMAE.csv'
                            y_predictions_tr = model.predict(X_train).tolist()
                            y_predictions_val = model.predict(X_val).tolist()
                            #print(y_predictions_tr)
                            output_predictions(y_train.tolist(), y_predictions_tr, y_val.tolist(), y_predictions_val, fil_predict_val, fil_predict_tr, comment)
                            best_mae = temp2
    
    
                    train_loss_av = np.mean(train_loss)
                    train_mae_av = np.mean(train_mae)
                    val_loss_av = np.mean(val_loss)
                    val_mae_av = np.mean(val_mae)
                    
                    #create list for outputs and send to save
                    outtodf = [train_loss_av, train_mae_av, val_loss_av, val_mae_av, db, lst, dus]
                    print(outtodf)
                    output_results(outtodf, fil_out)

# Ensure this block only runs when the script is executed directly
if __name__ == "__main__":
    main()
