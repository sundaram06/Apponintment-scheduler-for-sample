import pandas as pd
import streamlit as st
import numpy as np
import pickle

sample=pd.read_csv(r'E:\Pro 69\pro\final_data.csv')

st.title('Medical Sample Collection')

st.write(sample.head())
agent = st.selectbox("Select Agent", sample['Agent ID'].unique())

diagnostic = st.selectbox("Diagnostic Centers", sample['Diagnostic Centers'].unique())
if diagnostic == "Apollo Diagnostics":
    diagnostic = 0
elif diagnostic == "Diamond Diagnostic Center":
    diagnostic = 1
elif diagnostic == "Lucid Medical Diagnostics":
    diagnostic = 2
elif diagnostic == "Medifine Diagnostic Center":
    diagnostic = 3
elif diagnostic == "Medquest Diagnostics Center":
    diagnostic = 4
elif diagnostic == "Pronto Diagnostics Center":
    diagnostic = 5
elif diagnostic == "Sri Sai Diagnostic Center":
    diagnostic = 6
elif diagnostic == "Tesla Diagnostics":
    diagnostic = 7
elif diagnostic == "Vijaya Diagnostic Center":
    diagnostic = 8
else:
    diagnostic = 9

test = st.selectbox("Test name", sample['Test name'].unique())
if test == "Vitamin B-12":
    test = 0
elif test == "HbA1c":
    test = 1
elif test == "Vitamin D-25Hydroxy":
    test = 2
elif test == "TSH":
    test = 3
elif test == "Lipid Profile":
    test = 4
elif test == "Complete Urinalysis":
    test = 5
elif test == "RTPCR":
    test = 6
elif test == "H1N1":
    test = 7
elif test == "Fasting blood sugar":
    test = 8
else:
    test = 9

sample = st.selectbox("Sample", sample['Sample'].unique())
if sample == "Blood":
    sample = 0
elif sample == "Urine":
    sample = 1
else:
    sample = 2


dis_AP = st.number_input("shortest distance Agent-Pathlab(m)")

dis_PP = st.number_input('shortest distance Patient-Pathlab(m)')

dis_PA = st.number_input('shortest distance Patient-Agent(m)')

time_coll=st.number_input('Time For Sample Collection MM')    
time = st.number_input('Time Agent-Pathlab sec')
model = pickle.load(open('model.pkl','rb'))

if st.button('Predict'):

    re = np.array([agent,diagnostic,test,sample,dis_AP,dis_PP,dis_PA,time_coll,time])
    re = re.reshape(1,9)
 
    result=model.predict(re)
    st.header("Exact Arrival Time")
    st.write(str(result))
