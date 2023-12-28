import streamlit as vAR_st
import pandas as pd
import os
from openai import OpenAI



ASSISTANT_ID = os.environ["ASSISTANT_ID"]


def SpaceOptimization():
    
    client = OpenAI()
    # Delete Existing Assistant file 
    deleted_assistant_file = client.beta.assistants.files.delete(
    assistant_id=ASSISTANT_ID,
    file_id="file-abc123"
)
    print('deleted_assistant_file_response - ',deleted_assistant_file)
    
    # Step 1 - Read Dataset
    vAR_data = ReadFile()
    

    
        
    
    
    


def ReadFile():
    
    vAR_data = None
    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    with col2:
        vAR_st.subheader('Upload Training Data')
        
    with col4:
        vAR_dataset = vAR_st.file_uploader('Upload Dataset',type=["csv"])
        
        
        
    if vAR_dataset:
        
        # Uploda file into openai assistant
        bytes_data = vAR_dataset.getvalue()
        location = "source_data"
        with open(location,"wb") as f:
            f.write(bytes_data)
        
        if "file_id" not in vAR_st.session_state:
            vAR_st.session_state.file_id = []
            vAR_st.session_state.file_id = CreateAssistantFile(location)
            
        ############################################
        
        vAR_data = pd.read_csv(vAR_dataset)
        
        col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_preview = vAR_st.button("Preview Data")
            
        if vAR_preview:
            
            vAR_st.write("")
            vAR_st.write("")
            vAR_st.write("")
            vAR_st.write("")
            vAR_st.write("")
            vAR_st.dataframe(vAR_data)
            
        
        
    return vAR_data


def CreateAssistantFile(location):
    
    client = OpenAI()

    file = client.beta.assistants.files.create(
    file=open(location,'rb'),purpose="assistants"
    )
    # os.remove(location)
    print('assistant_file - ',file)
    
    return file.id
    
