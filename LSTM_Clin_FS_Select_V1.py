#Code to set up classification model for UPDRS level based on a number of clinical features for data from the PPMI data set 
#code runs LSTM with 3 layers which can have defined number of nodes in each Layer
#2 * LSTM Layer
#1 * Dense Layer
#Features are divided into a number of feature sets which can be used individiually or combined
# The number of features varies acorss the features set, a cut off set is to allow features to be 95% complet and then for N/A's to be filled with the previous value
#4 Classes are set for UPDRS levels based on even quartiles from the data set used

#import relevant modules
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import array, quantile
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Masking
from keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

#function to load fsindex
def fsind_load (fil_path):
    df = pd.read_csv(fil_path)
    return df

#load feature set function
def load_featuresets(FS_list, FS_df):
    #check of FS i.e. ID fields for data set
    if 'FS1' not in FS_list:
        FS_list.insert(0,'FS1')      
    #make sure the ID fields i.e FS1 are first in line
    if FS_list.index('FS1') != 0:
        FS_list.pop(FS_list.index('FS1'))
        FS_list.insert(0,'FS1')      
    
    #fs_count = len(FS_list)
    #find list of files that we need to load
    fs_filelst = FS_df['FS_File'][FS_df['FS_NO'].isin(FS_list)].drop_duplicates().to_list()
    selected_features = FS_df['FS_Name'][FS_df['FS_NO'].isin(FS_list)].to_list()
    #print(selected_features)
    #print(FS_list)
    #print(fs_filelst)
    
    df_count = 0
    df_list = []
    #load files into list of df's
    for fil in fs_filelst:
        df_list.append(pd.read_csv(fil))
        df_count += 1

    #select desired feature groups from the first df
    col_list_df1 = df_list[0].columns.to_list()
    #print(col_list_df1)
    #drop columns from not desired feature groups
    for col in col_list_df1:
        if not col.split('.')[0] in selected_features:
            df_list[0].drop([col], axis=1, inplace = True)

    #remove FS category header from first df and replace with columns names from row 1
    df_list[0] = df_list[0].rename(columns=df_list[0].iloc[0]).drop(df_list[0].index[0])

    #capture list of patno's across the feature sets/df's
    patno_lst = []
    for i in range(len(df_list)):
        patno_lst.append(df_list[i]['PATNO'].to_list())

    #find common PATNO's across the lists
    sets = map(set, patno_lst)
    common = set.intersection(*sets)
    patno_lst = list(common)

    #select df's data whihc is on common patno list
    for i in range(len(df_list)):
        df_list[i] = df_list[i][df_list[i]['PATNO'].isin(patno_lst)]

    
    #concatmethod
    #selected_df = pd.concat(df_list)
    #merge method
    selected_df = df_list[0][df_list[0]['PATNO'].isin(patno_lst)]
    for i in range(len(df_list)-1):
        #selected_df = selected_df.merge(df_list[i + 1], on = ['PATNO','EVENT_ID'])#inner
        #selected_df = selected_df.merge(df_list[i + 1], on = ['PATNO','EVENT_ID'], how = 'outer')#outer
        selected_df = selected_df.merge(df_list[i + 1], on = ['PATNO','EVENT_ID'], how = 'left')#left inner
        #selected_df = selected_df.merge(df_list[i + 1], on = ['PATNO','EVENT_ID'], how = 'right')#right inner

    return selected_df

#method to select required basic ID data and UPDRS score
def prepareid_UPDRS(df_in):
    # Drop rows where we have no UPDRS off total score!
    df_out = df_in.dropna(subset=['updrs_totscore'])
    #keep key columns
    cols_to_keep = ['PATNO', 'EVENT_ID','YEAR','subgroup','age_at_visit','SEX','updrs_totscore']
    df_out = df_out[cols_to_keep]

    return df_out

#function to select partipanats with given number of consecutive visits expressed as an integer list
def get_patdata(df, yr_list):
    #get all rows with relevant events
    filtered_df = df[df['YEAR'].isin(yr_list)]
    #count per patnumber
    value_counts = filtered_df['PATNO'].value_counts()
    # Get values that meet the threshold
    valid_values = value_counts[value_counts == len(yr_list)].index  
    #Filter the DataFrame to keep only rows where 'PATNO' is in the valid_values
    filtered_df = filtered_df[filtered_df['PATNO'].isin(valid_values)]
    
    filtered_df = filtered_df.sort_values(by= ['PATNO','YEAR'])
    return filtered_df


#select athcing features
def preparefs_match(df_in1, df_in2):
    #Select only the columns whihc are not in the ID data and add patno and event ID to join
    remaining_columns = df_in1.columns.difference(df_in2.columns).to_list()
    #add back the two key columns for joining
    remaining_columns.insert(0,'EVENT_ID')
    remaining_columns.insert(0,'PATNO')
    #select columns from feature set
    df_out = df_in1[remaining_columns]
    return df_out


def merge_dumpna(df_in1, df_in2, thres):
    #merge data frames
    df_out = df_in1.merge(df_in2, on = ['PATNO','EVENT_ID'], how = 'left')#left inner

    # Identify columns to drop
    columns_to_drop = df_out.columns[df_out.isna().mean() > thres]

    # Drop the columns
    df_out = df_out.drop(columns=columns_to_drop)
    
    return df_out


def namethod_fill(df_in):
    #fill na's with previous rows values
    df_out = df_in.fillna(method='bfill')
    return df_out

#function to load the data
def load_data (df):
    
    #get counts of patients & visits
    patient_labels = df['updrs_totscore'].to_list()
    visits = len(df['EVENT_ID'].unique().tolist())
    sample_ct = len(df['PATNO'].unique().tolist())
    
    #drop data info columns
    
    df.drop(df.columns[[0,1,2,3,6]], axis = 1,inplace=True)
    
    #get feature size
    features = df.shape[1]

    # Flatten the list
    #patient_labels = [item for sublist in patient_labels for item in sublist]
    # add back patient labels
        # add back patient labels
    df.insert(0, 'updrs_totscore', patient_labels)
    return df, visits, features, sample_ct

#shape the data for LSTM modells
def shape_data(input_df, visits_st, sample_ct, num_cat):
    sequences_temp = []
    patient_labels = []
    labels_temp = []
    quantiles = []
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

        #scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values)
        sequences_temp.append(scaled_data)
        labels_temp.append(patient_labels)

    #shape sequence array for LSTM (samples, timeseries, fetures) format
    sequences_temp = array(sequences_temp).reshape(sample_ct, visits_act, features_act)

    #set up list of quantiles for ranges
    for i in range(num_cat - 1):
         quantiles.append(np.quantile(array(labels_temp), (i+1)/num_cat))
     
    #define categories based on dividing max value by Max UPDRS/num of categories 
    for i in range(len(labels_temp)):
         for q in reversed(range(len(quantiles))):
               if labels_temp[i] > quantiles[q]:
                    labels_temp[i] = q + 1
                    break
               elif labels_temp[i] <= quantiles[0]:
                    labels_temp[i] = 0
     
    # convert labels to one hot encoding
    labels_temp = array(labels_temp)
    #print(labels_temp)
    labels_temp = to_categorical(labels_temp)
        
    return sequences_temp, labels_temp, visits_act


#define LSTM model
def model_define(input_shape, num_cat, model_structure):
    #input_shape=(numof_outputs, numof_features)
# define model where LSTM is also output layer
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(model_structure[0],dropout=0.2,return_sequences=True))
    model.add(LSTM(model_structure[1],dropout=0.2))
    model.add(Dense(model_structure[2],activation='relu'))
    model.add(Dense(num_cat,activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return(model)

#set up results output file
def output_results(outlist, outfile):
    out_df = pd.read_csv(outfile)
    new_row_df = pd.DataFrame([outlist], columns=out_df.columns)
    out_df = pd.concat([out_df, new_row_df], ignore_index=True)
    out_df.to_csv(outfile, index=False)

#main
def main():

    #1a)load feature set file
    fs_file = 'Feature_SetIndex.csv'
    fets_df = fsind_load(fs_file)

    #1b) 
    #select ID's demographics and UPDRS
    select_list = ['FS1','FS2','FS8']
    #collect feature
    SELECTION_df = load_featuresets(select_list, fets_df)
    updrs_df = prepareid_UPDRS(SELECTION_df)
    # List of event IDs to filter
    event_testlist = [0,1,2,4]  
    #get selected event list only
    updrs_df = get_patdata(updrs_df,event_testlist)
    #select a feature set alongside ID information to enable matching
    #collect feature set and match to ID's and events
    fs_list = ['FS1','FS3']
    fs_df = load_featuresets(fs_list, fets_df)
    fs1_df = load_featuresets(['FS1'], fets_df)
    fs_match = preparefs_match(fs_df, fs1_df)
    #merge feature set and eliminate all feature columns which have number %of Na greater than threshold
    new_df = merge_dumpna(updrs_df, fs_match, 0.05)

    new_df = namethod_fill(new_df)

    print(event_testlist)
    print(new_df.shape)
    print(new_df.shape[0]/len(event_testlist))

    q1 = new_df['updrs_totscore'].quantile(0.25)  # First quartile (25th percentile)
    q2 = new_df['updrs_totscore'].quantile(0.50)  # Second quartile (50th percentile / Median)
    q3 = new_df['updrs_totscore'].quantile(0.75)  # Third quartile (75th percentile)

    print("First Quartile (Q1):", q1)
    print("Second Quartile (Median/Q2):", q2)
    print("Third Quartile (Q3):", q3)


    #Steps to implement
    #1) send data from to 'load method' to format it
    fullfeature_df, visits_total, features_total, sample_ct = load_data (new_df)
    print(features_total)
    print(visits_total)
    print(sample_ct)
    #create file for out put
    fil_out = "_".join(fs_list) + '_' +"_".join(map(str, event_testlist)) + '_output.csv'
    outtodf = ['train_loss', 'train_acc', 'test_loss', 'test_acc', 'L1', 'L2', 'L3', 'q1','q2','q3','Visit List', 'Features','Samples']
    df = pd.DataFrame(columns=outtodf)
    df.to_csv(fil_out, index=False)
    
    #2) shape the data
    num_cat = 4 #split data into 4 even quantiles
    sequences, labels, visits_red = shape_data(fullfeature_df, visits_total, sample_ct, num_cat)
    print(sequences.shape)
    print(labels.shape)
    print(visits_red)

    
    #3)set up training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    #4) create the model
    # Define input shape
    input_shape = (visits_red, features_total + 1)

    #cycle list of variables
    #L1 = [8,16,32,64,128,256]
    #L2 = [8,16,32,64,128,256]
    #L3 = [8,16,32,64,128,256]
    
    #testing cycle list
    L1 = [64]
    L2 = [32]
    L3 = [32]



    for A1 in L1:
        for B1 in L2:
            for C1 in L3:
                # Build the model
                model = model_define(input_shape, num_cat, [A1,B1,C1])

                #5) Train the model
                # Train the model
                X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42, stratify=labels)

                history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose = False)
                #6) output the data from the model
                #function to plot graphs if required
                #plot_tranhist(history)
                # Evaluate the model on test data
                test_loss, test_acc = model.evaluate(X_val, y_val)
                #print(f"Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")
                train_loss, train_acc = model.evaluate(X_train, y_train)
                #print(f"Train Loss (MSE): {train_loss:.4f}, Train MAE: {train_mae:.4f}")
                #create list for outputs and send to save
                outtodf = [train_loss, train_acc, test_loss, test_acc, A1, B1, C1,q1,q2,q3,len(event_testlist),features_total, sample_ct]
                print(outtodf)
                output_results(outtodf,fil_out)

# Ensure this block only runs when the script is executed directly
if __name__ == "__main__":
    main()
