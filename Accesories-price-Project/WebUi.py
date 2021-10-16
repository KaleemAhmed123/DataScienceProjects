import streamlit as st
import pickle
import numpy as np

# Loading the model here
data = pickle.load(open('data.pkl', 'rb'))
pipeline = pickle.load(open('pipe.pkl', 'rb'))

# Main title
st.title("Accesories Price Predictor ")

# Select option for brand name
brand = st.selectbox("Brand-name", data['Company'].unique())

# Type selection
pcType = st.selectbox("Type ", data['TypeName'].unique())

# Ram selection
X = data['Ram'].unique()
X.sort()
ram = st.selectbox("Ram (GB)", X)

# Weight selection
weight = st.number_input("Weight of the Laptop (KG)",
                         min_value=0.7, max_value=4.70, step=0.1, value=0.7)


# Touchscreen bool
touchscreen = st.selectbox("TouchScreen ", ['No', 'Yes'])

# Ips Panel selection
ips_panel = st.selectbox("Ips-Panel ", ['No', 'Yes'])

# Screen Size selection
screen_size = st.number_input(
    'Screen Size (Inches)', min_value=10.00, max_value=18.50, value=10.00, step=0.1)

# ScreenResolution selection
resolution = st.selectbox('Screen Resolution',
                          ['1366x768', '1920x1080', '2304x1440',
                           '1600x900', '3840x2160', '3200x1800',
                           '2880x1800', '2560x1600', '2560x1440'])

# Cpu Brand selection
cpu = st.selectbox("Cpu ", data['Cpu Brand'].unique())

# HDD selection
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])

# SDD selection
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])

# Gpu Brand selection
gpu = st.selectbox("Gpu ", data['Gpu Brand'].unique())

os = st.selectbox("Os ", data['OS Category'].unique())


if st.button("Predict Price "):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips_panel == 'Yes':
        ips_panel = 1
    else:
        ips_panel = 0

    xResolution = int(resolution.split('x')[0])
    yResolution = int(resolution.split('x')[1])

    ppi = ((xResolution**2) + (yResolution**2))**0.5/screen_size

    query = np.array([brand, pcType, ram, weight, touchscreen,
                      ips_panel, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    finalOutcome = int(np.exp(pipeline.predict(query)[0]))

    st.title("The predicted price of this configuration is around ......")
    st.title(str(finalOutcome) + " INR")
    st.write("or ")
    st.title(str(finalOutcome//76) + " $")
