import streamlit as vAR_st
import pandas as pd
import os
from openai import OpenAI
from DSAI_SourceCode_Implementation.DSAI_SpaceOptimize.DSAI_AssistantAPI import DataInsightsWithAssistant
import base64
from PIL import Image
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,LabelEncoder
from sklearn.metrics import accuracy_score
from bokeh.models.widgets import Div

PUMP_ASSISTANT_ID = os.environ["PUMP_ASSISTANT_ID"]


def WaterPumpMaintenance():
    
    if 'vAR_preview_maintenance' not in vAR_st.session_state:
        vAR_st.session_state.vAR_preview_maintenance = False
        
    if 'vAR_model_train_maintenance' not in vAR_st.session_state:
        vAR_st.session_state.vAR_model_train_maintenance = False
        
    if 'vAR_preprocessed_data_maintenance' not in vAR_st.session_state:
        vAR_st.session_state.vAR_preprocessed_data_maintenance = None
        
    if 'vAR_model_test_maintenance' not in vAR_st.session_state:
        vAR_st.session_state.vAR_model_test_maintenance = False
        
    if 'vAR_model_maintenance' not in vAR_st.session_state:
        vAR_st.session_state.vAR_model_maintenance = None
        
    if "vAR_model_outcome_maintenance" not in vAR_st.session_state:
        vAR_st.session_state.vAR_model_outcome_maintenance = pd.DataFrame()
    
    if "flag" not in vAR_st.session_state:
        vAR_st.session_state.flag_maintenance = False
        
    
        
    # Step 1 - Read data
    vAR_data = ReadFile()
    
    if vAR_st.session_state.vAR_preview_maintenance:
        # Step 2 - Stats Report
        # StatsReport(vAR_data)
        
        # Step 3 - Extract Insights with LLM
        ExtractInsightsWithLLM()
        
        # Step 4 - Data Preprocessing
        # To handle null values,outliers and feature extraction
        # vAR_data = DataPreprocessing(vAR_data)
       
    # Step 5 - Feature Selection                 
    # if vAR_st.session_state.vAR_data_preprocess:
    #     FeatureSelection(vAR_st.session_state.vAR_preprocessed_data)
        
        # Step 6 - Model Training
        # if len(vAR_st.session_state.vAR_features)>0:
        vAR_st.session_state.vAR_preprocessed_data_maintenance = vAR_data
        ModelTraining(vAR_st.session_state.vAR_preprocessed_data_maintenance)
        
    # Step 7 - Model Testing
    if vAR_st.session_state.vAR_model_train_maintenance:
        vAR_model_outcome = ModelTesting(vAR_st.session_state.vAR_model_maintenance)
        
        vAR_st.write("")
        vAR_st.write("")
        col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
        with col2:
            vAR_st.dataframe(vAR_model_outcome)   
        
    if vAR_st.session_state.vAR_model_test_maintenance and len(vAR_st.session_state.vAR_model_outcome_maintenance)>0:
        vAR_csv = vAR_st.session_state.vAR_model_outcome_maintenance.to_csv(index=False).encode('utf-8')
        vAR_st.session_state.flag_maintenance = True
        col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
        with col4:
            vAR_st.download_button(
    "Download Model Outcome",
    vAR_csv,
    "model_output.csv",
    "text/csv",
    key='download-csv'
    )
        # # Step 8 - Model Outcome Visualization & Download
        # if len(vAR_st.session_state.vAR_model_outcome)>0 and vAR_st.session_state.flag:
        #     ModelOutcomeVisuals(vAR_model_outcome)
            
        # Step 9 - Looker studio Reports
        
        # if len(vAR_st.session_state.vAR_model_outcome)>0 and vAR_st.session_state.flag:  
        #     ExploreWithLooker()

            

    
        
    
    
    


def ReadFile():
    
    vAR_data = None
    vAR_preview = False
    vAR_dataset = None
    
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Select Data source")
        
        
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_choice = vAR_st.selectbox(" ",["Select the source","Sensor Device","Surveillance Camera","Video Streaming","Database","File Upload"])
    
    if vAR_choice=="File Upload":
        col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Upload Training Data')
            
        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_dataset = vAR_st.file_uploader('Upload Dataset',key="water-train-file")
        
        
        
    if vAR_dataset:
        
        vAR_data = pd.read_csv(vAR_dataset)
        
        col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
        with col4:
            vAR_st.write('')
            vAR_preview = vAR_st.button("Preview Data")
            
        if vAR_preview:
            
            col1,col2,col3 = vAR_st.columns([2,8,2])
            with col2:
                vAR_st.session_state.vAR_preview_maintenance = True
                vAR_st.write("")
                vAR_st.dataframe(vAR_data)
            
        
    return vAR_data


def StatsReport(vAR_data):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Stats Report')
        
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_stats_report = vAR_st.button('Get Stats Report')
    
    if vAR_stats_report:
        vAR_st.session_state.vAR_stats_report = True
        vAR_st.write('')
        vAR_st.write('')
        col1,col2,col3 = vAR_st.columns([2,8,2])
        with col2:
            vAR_st.dataframe(vAR_data.describe())
            
        
        


def DataPreprocessing(vAR_data):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Data Preprocessing')
        
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_stats_report = vAR_st.button('Data Preprocessing')
    
    if vAR_stats_report:
        vAR_st.session_state.vAR_data_preprocess = True
    
    
        vAR_data['Event'] = vAR_data['Event'].fillna(vAR_data['Event'].mode()[0])
        vAR_data['SpecialEquipment'] = vAR_data['SpecialEquipment'].fillna(vAR_data['SpecialEquipment'].mode()[0])
        
        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.success("Data Preprocessing completed!")
        
        vAR_st.session_state.vAR_preprocessed_data = vAR_data
            
    return vAR_data
    
    
def FeatureSelection(vAR_data):
    
    vAR_features_list = list(vAR_data.columns)[:-1]
    vAR_features_list.append("All")
    print(list(vAR_data.columns))
    vAR_features_list = vAR_features_list
    
    print('list - ',vAR_features_list)
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Select Features')
        
    with col4:
        vAR_st.write('')
        vAR_features = vAR_st.multiselect(' ',vAR_features_list,default="All")
    
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    with col2:
        vAR_st.write('')
        with vAR_st.expander("List selected features"):  
            if 'All' in vAR_features:
                vAR_st.write('Features:',vAR_features_list[1:])
            else:
                for i in range(0,len(vAR_features)):
                    vAR_st.write('Feature',i+1,':',vAR_features_list[i])
    vAR_st.session_state.vAR_features = vAR_features
    return vAR_features
    

def ModelTraining(vAR_data):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Model Training')
        
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_train = vAR_st.button('Train the Model')
    
        if vAR_train:
            vAR_st.session_state.vAR_model_train_maintenance = True
            # Applying OneHotEncoder
            vAR_X = vAR_data.drop(columns = ["Failure Indicator","Timestamp", "Pump ID"])
            vAR_y = vAR_data["Failure Indicator"]
            
            label_encoder = LabelEncoder()
            vAR_X['Maintenance History'] = label_encoder.fit_transform(vAR_X['Maintenance History'])
            
            # Initializing MinMaxScaler
            scaler = MinMaxScaler()

            # Applying scaler to the combined data
            scaled_data = scaler.fit_transform(vAR_X)
            
            print('scaled data shape - ',scaled_data.shape)
            
            print('scaled data - ',scaled_data[:10])
            
            # Splitting the data
            # X_train, X_test, y_train, y_test = train_test_split(scaled_data, vAR_y, test_size=0.2, random_state=42)
            
            X_train = scaled_data
            y_train = vAR_y
            
            # Model Selection
            model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Model Training
            model.fit(X_train, y_train)
            
            vAR_st.session_state.vAR_model_maintenance = model
            
            vAR_st.success("Model successfully trained!")
                
def ModelTesting(vAR_model): 
    vAR_test_model = False
    vAR_X = pd.DataFrame()
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Upload Test Data')
        
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_test_data = vAR_st.file_uploader('Upload Test Data',key="water-test-file")
        if vAR_test_data:
            vAR_test_data = pd.read_csv(vAR_test_data)
            vAR_st.write('')
            vAR_st.write('')
            vAR_test_model = vAR_st.button("Test Model")
        
    if vAR_test_model:
        vAR_st.session_state.vAR_model_test_maintenance = True
        
        with vAR_st.spinner("Model Testing is in-progress"):
            
            
            vAR_X = vAR_test_data
            
            label_encoder = LabelEncoder()
            vAR_test_data['Maintenance History'] = label_encoder.fit_transform(vAR_test_data['Maintenance History'])
            
            
            # Initializing MinMaxScaler
            scaler = MinMaxScaler()

            # Applying scaler to the combined data
            scaled_data = scaler.fit_transform(vAR_test_data)
            
            print('scaled data shape - ',scaled_data.shape)
            
            print('scaled data - ',scaled_data[:10])
            
            y_pred = vAR_model.predict(scaled_data)
            
            print('y pred - ',y_pred)
            
            vAR_X['Prediction'] = ['Likely to Fail' if pred == 1 else 'Likely Safe' for pred in y_pred]
            
            vAR_st.session_state.vAR_model_outcome_maintenance = vAR_X
            
            return vAR_X
    
    
    









def ExtractInsightsWithLLM():
    vAR_llm_response = None
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Interact with LLM')
        
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_user_input = vAR_st.text_area('Sample : Which pump will be defective first? Why?')

        vAR_submit = vAR_st.button("Submit")
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    if vAR_submit:   
        if vAR_user_input:
            vAR_llm_response = DataInsightsWithAssistant(vAR_user_input,PUMP_ASSISTANT_ID)
            
            with col2:
                vAR_st.info(vAR_llm_response) 
            
        # string_bytes = vAR_llm_response.encode("utf-8")
        # base64_string = base64.b64encode(string_bytes)
        
        # vAR_decoded_data = base64.b64decode(base64_string)
        
        # vAR_st.image(Image.open(io.BytesIO(vAR_decoded_data)))
        # ----------------------------------------------------
        # with open("llm_image.png","w") as f:
        #     f.write(base64.b64encode(vAR_llm_response))
            
        # vAR_st.image("llm_image.png")
        # -------------------------------------------------------
        # image_bytes = bytes(vAR_llm_response, 'latin-1')  # 'latin-1' encoding maps byte values directly to Unicode

        # # You might need to save this data to a file and then read it back, depending on the format
        # with open('temp_image.png', 'wb') as file:
        #     file.write(image_bytes)

        # vAR_st.image("temp_image.png")
        
        # -------------------------------------------------------------
        
        # Convert the string to bytes - assuming the string is a byte array representation
        # image_bytes = vAR_llm_response.encode('utf-8', 'backslashreplace')
        # image_bytes = image_bytes.replace(b"\\x", b"")
        # image_bytes = bytes.fromhex(image_bytes.decode('utf-8'))

        # # Create an image from bytes
        # image = Image.open(io.BytesIO(image_bytes))
        # vAR_st.image(image)
    
    return vAR_llm_response
    