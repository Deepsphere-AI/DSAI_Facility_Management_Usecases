import streamlit as vAR_st
from openai import OpenAI
import pandas as pd


def LayoutSuggestion():


    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    col6,col7,col8 = vAR_st.columns([2,10,2])
    col9,col10,col11,col12,col13 = vAR_st.columns([2.2,9,0.7,10,2])

    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Generate Layout/Data")
        
        
        
    with col4:
        vAR_st.write('')
        vAR_choice = vAR_st.selectbox(" ",["Select Datatype","Generate Facility Layout Image","Generate Facility Layout Data"])

        

    if vAR_choice!="Select Datatype":

        with col10:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader("Location/Postal Code")

        with col12:
            vAR_st.write('')
            vAR_postal_choice = vAR_st.selectbox(" ",["Select PostalCode","608526","521147","546080","569933","519355"])



    if vAR_choice=="Generate Facility Layout Image":
        GenerateLayoutImage()
    
    elif vAR_choice!="Select Datatype" and vAR_choice=="Generate Facility Layout Data":
        with col7:
            vAR_st.write("")
            vAR_st.write("")
            vAR_st.info("Work in-progress")

def GenerateLayoutImage():

    col1,col2,col3,col4,col5 = vAR_st.columns([2.2,9,0.7,10,2])
    col9,col10,col11,col12,col13 = vAR_st.columns([2.2,9,0.7,10,2])

    col6,col7,col8 = vAR_st.columns([2,16,2])

    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Facility Type")

        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Facility Specifications")
        
        
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_facility_type = vAR_st.selectbox(" ",["Select Anyone","Corporate Office","Shopping Mall","Hospital","Restaurant","Factory"])

        vAR_st.write('')
        vAR_spec_file = vAR_st.file_uploader('Specification File')

    if vAR_spec_file:
        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_facility_spec_preview = vAR_st.button("Preview Facility Specification")
        with col7:
            if vAR_facility_spec_preview:
                vAR_spec = pd.read_csv(vAR_spec_file)
                vAR_st.dataframe(vAR_spec)
                

           


    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_facility_type_submit = vAR_st.button("Submit")

    if vAR_facility_type_submit and vAR_facility_type!="Select Anyone":
        client = OpenAI()
        

        # with col7:

        with vAR_st.spinner("Generating Layout and it's details.."):
            image_response = client.images.generate(
            model="dall-e-3",
            prompt=f"""
            Generate an Image for below scenario:

        I have my commercial real estate area with overall 2000 sqft in rectangle shape for {vAR_facility_type} usage. I would like to create layout design for my facility with all areas including work area, rest rooms, reception, inventory area,emergency exits, HVAC systems, etc. Suggest me a proper layout design based on the type of service.And, mark different colors for each area. 

        Also, mark each area name and it's square feet in the image itself.

        And Give me the details of how many members can occupy in each area.Mention the details of color and area name in the seperate text response. Make sure the texts should be clear in image.
            """,
            size="1024x1024",
            quality="standard",
            n=1
            )
            
            print('image response - ',image_response)

            image_url = image_response.data[0].url

            print("image_url- ",image_url)

            vAR_st.image(image_url)


            vAR_st.write('')
            vAR_st.write('')
            response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                "role": "user",
                "content": f"""
                Give me layout suggestion plan for below scenario:\nI have my commercial real estate area with overall 2000 sqft in rectangle shape for {vAR_facility_type} usage. How can i allocate my facility with all areas including work area, rest rooms, reception, inventory area,emergency exits, HVAC systems, etc. Suggest me a proper plan.
                    Also, draw rough sketch of how the layout might look with area size in sqft and without any empty space in it.
                """
                }
            ],
            temperature=0,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )   

            vAR_st.info(response.choices[0].message.content)

            

