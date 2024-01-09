import streamlit as vAR_st
import pandas as pd
import os
from DSAI_SourceCode_Implementation.DSAI_SpaceOptimize.DSAI_AssistantAPI import DataInsightsWithAssistant

INVENTORY_ASSISTANT_ID = os.environ["INVENTORY_ASSISTANT_ID"]

def InventoryManagement():

    if 'vAR_preview_inventory' not in vAR_st.session_state:
        vAR_st.session_state.vAR_preview_inventory = False

    # Step 1 - Read data
    vAR_data = ReadFile()

    if vAR_st.session_state.vAR_preview_inventory:
        ExtractInsightsWithLLM()



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
        vAR_choice = vAR_st.selectbox(" ",["Select the source","Live Camera","Video","File Upload"])
    
    if vAR_choice=="File Upload":
        col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Upload Training Data')
            
        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_dataset = vAR_st.file_uploader('Upload Dataset',key="inventory-file")
        
        
        
    if vAR_dataset:
        
        vAR_data = pd.read_csv(vAR_dataset)
        
        col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
        with col4:
            vAR_st.write('')
            vAR_preview = vAR_st.button("Preview Data")
            
        if vAR_preview:
            
            col1,col2,col3 = vAR_st.columns([2,8,2])
            with col2:
                vAR_st.session_state.vAR_preview_inventory = True
                vAR_st.write("")
                vAR_st.dataframe(vAR_data)
            
        
    return vAR_data





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
        vAR_user_input = vAR_st.text_area('Sample : What is the number of assets in each category.')

        vAR_submit = vAR_st.button("Submit")
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    if vAR_submit:   
        if vAR_user_input:
            vAR_llm_response = DataInsightsWithAssistant(vAR_user_input,INVENTORY_ASSISTANT_ID)
            
            with col2:
                vAR_st.info(vAR_llm_response) 
    
