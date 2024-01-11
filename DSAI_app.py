import streamlit as vAR_st
vAR_st.set_page_config(page_title="Facility Management", layout="wide")

from DSAI_Utility.DSAI_Utility import All_Initialization,CSS_Property
from DSAI_SourceCode_Implementation.DSAI_SpaceOptimize.DSAI_SpaceOptimize import SpaceOptimization
from DSAI_SourceCode_Implementation.DSAI_LayoutSuggestion.DSAI_LayoutSuggestion import LayoutSuggestion
from DSAI_SourceCode_Implementation.DSAI_InventoryManagement.DSAI_InventoryManagement import InventoryManagement

from DSAI_SourceCode_Implementation.DSAI_WaterPump_Maintenance.DSAI_Maintenance import WaterPumpMaintenance

import traceback



if __name__=='__main__':
    vAR_hide_footer = """<style>
            footer {visibility: hidden;}
            </style>
            """
    vAR_st.markdown(vAR_hide_footer, unsafe_allow_html=True)
    try:
        # Applying CSS properties for web page
        CSS_Property("DSAI_Utility/DSAI_style.css")
        # Initializing Basic Componentes of Web Page
        choice = All_Initialization()

        if choice=='Facility Management Usecases':
            # Calling ChatGPT
            
            col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
            with col2:
                vAR_st.write('')
                vAR_st.subheader('Select Usecase')
                
                
            with col4:

                vAR_vendor = vAR_st.selectbox(' ',["Select Usecase","Space Optimization","Building Layout Recommendation","Asset & Inventory Management","Predictive Maintenance(WaterPump)"])
            
            
            if vAR_vendor=='Space Optimization':
                SpaceOptimization()
        
            elif vAR_vendor=="Building Layout Recommendation":
                LayoutSuggestion()
            elif vAR_vendor=="Asset & Inventory Management":
                InventoryManagement()
            elif vAR_vendor=="Predictive Maintenance(WaterPump)":
                WaterPumpMaintenance()
            else:
                pass


    except BaseException as exception:
        print('Error in main function - ', exception)
        exception = 'Something went wrong - '+str(exception)
        traceback.print_exc()
        vAR_st.error(exception)
