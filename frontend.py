import streamlit as st
import requests

# Set up page
st.set_page_config(page_title="GRAILS | BiasAI", layout="centered")

st.title("AI Bias Detector")
st.write("Upload a dataset to detect hidden algorithmic bias.")

# Widget of downloading file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    st.success("File selected! Ready for analysis.")
    
    # Start button 
    if st.button("Run Bias Audit", type="primary"):
        with st.spinner("Sending data to the backend..."):
            
            # Sending faile on FastAPI to our backend.py
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            
            try:
                # Addres of local server backend.py
                response = requests.post("http://127.0.0.1:8000/api/analyze", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.write("### Backend Response:")
                    st.json(data) # Output of FastAPI 
                    
                    # Here we need to draw graphs 
                    st.write("### Dashboard (Mock Area)")
                    st.info("Sahar,you can build fake Streamlit charts here while we wait for Luca's logic!")
                    
                else:
                    st.error(f"Backend error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure Anton's FastAPI is running!, ping me")
