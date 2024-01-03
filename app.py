import streamlit as st
import pickle
import numpy as np
import math



def main():

    # Importing the Model
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))

    st.title(":red[Laptop Price Predictor Virendra]")

    # Brand
    # company = st.selectbox('Brand', df['Company'].unique())
    defaultCompany = df['Company'].unique()[1]
    company = st.selectbox('Brand', df['Company'].unique(), index=df['Company'].unique().tolist().index(defaultCompany))

    

    # type of laptop
    # type = st.selectbox('Type', df['TypeName'].unique())
    defaultType = df['TypeName'].unique()[1]
    type = st.selectbox('Type', df['TypeName'].unique(), index=df['TypeName'].unique().tolist().index(defaultType))
    

    # Ram
    # ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])
    defaultRam = 8
    availableRam = [2,4,6,8,12,16,24,32,64]
    ram = st.selectbox('RAM (in GB)', availableRam, index = availableRam.index(defaultRam))

    # Weight
    weight = st.number_input('Weight of Laptop', value=1.4)

    # Touchscreen
    touchscreen = st.selectbox('TouchScreen',  ['No', 'Yes'])

    # IPS
    ips = st.selectbox('IPS',  ['No', 'Yes'])


    # Screen Size
    screen_size = st.number_input('Screen Size (in inch)', value=14)

    # resolution
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768',    '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440',   '2304x1440'])


    # cpu
    cpu = st.selectbox('CPU',  df['Cpu Brand'].unique())

    # HDD
    hdd = st.selectbox('HDD (in GB)',  [0, 128, 256, 512, 1024, 2048])

    # SSD
    # ssd = st.selectbox('SSD (in GB)',  [0, 128, 256, 512, 1024])
    availableSSD = [0, 128, 256, 512, 1024]
    defaultSSD = 512
    ssd = st.selectbox('SSD (in GB)', availableSSD, index = availableSSD.index(defaultSSD))

    # GPU
    gpu = st.selectbox('GPU',  df['Gpu brand'].unique())

    # OS
    # os = st.selectbox('OS',  df['os'].unique())
    defaultOS = df['os'].unique()[2]
    os = st.selectbox('OS', df['os'].unique(), index=df['os'].unique().tolist().index(defaultOS))

    if st.button('Predict Price'):

        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

        query = np.array([company, type, ram, weight, touchscreen, ips, ppi,    cpu, hdd, ssd, gpu, os])

        query = query.reshape(1, 12)
        rslt = math.floor(int(np.exp(pipe.predict(query)[0])))

        # st.title("The Predicted Price of Laptop is: "+str(rslt))
        st.title(f"The Predicted Price of Laptop is: :blue[{rslt}] :red[ \u20B9]")


if __name__=='__main__':
    main()



