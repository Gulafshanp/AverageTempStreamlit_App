#Importing Libraries

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
import geocoder
#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from http import server
server.maxMessageSize = 1000
st.set_option('deprecation.showPyplotGlobalUse', False)

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go



# For time stamps
from datetime import datetime

import plotly.express as px
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import requests
from io import StringIO
from zipfile import ZipFile, BadZipFile
from io import BytesIO
import pandas


## Importing Datasets

lon_lat = pd.read_csv("https://raw.githubusercontent.com/Gulafshanp/DataAnalyticsproj_Dataset/main/Longitude_Latitude.csv")
marine = pd.read_csv("https://raw.githubusercontent.com/Gulafshanp/DataAnalyticsproj_Dataset/main/Global_2020_MarineSpeciesRichness_AquaMaps.csv")


#### Here Temp Dataset which is Temperature dataset will be given more
## Priority as We will be predicting the avg temperature
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/kaggle-data-sets/29/2150/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240121%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240121T085057Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4a0227dbfdc523ea2ff50bba825702e9a198e45d096f22f9ef3e070ec33d7aa1fa8379764e1d82d7cfd3897c1ca0ff03e415f65f0d10c7a3c0bb003b2d91af325ed5d25ec3d017a559adac1cb5f44c2f124ba7b8c81e770b99f47b9cc18ca9f28a75f0cff17521383026c82603fe237d3eccdbb5a7b08fe528180277e3117dfc7d823b15371c1ab6a22214782fb96f2d425ff880458de08c926505f3a113c30dc5875ddb7635c593a42cfe05f61b40a7d6243d4df7114095e3b02856b2a4ce5f4e22e00e21f313867e447b488d42c5ec86c15520c0ff1b279928e41d281a66e2f65cbef309255be7f4d5d1a1b2b21136643a5af51b27892d30a20a8a2cb72795"
    content = requests.get(url)
    zf = ZipFile(BytesIO(content.content))

    for item in zf.namelist():
        print("File in zip: "+  item)

    # find the first matching csv file in the zip:
    match = [s for s in zf.namelist() if ".csv" in s][0]
    # the first line of the file contains a string - that line shall de     ignored, hence skiprows
    global temp
    temp = pandas.read_csv(zf.open(match), low_memory=False)
    return temp


temp = load_data()
#---------------------------------------------------------------#
def main():
    st.title("Average Temperature Forecasting App")
    st.subheader("""Access The Menu/ Options:
                    At Upper Left Corner in The Sidebar!!😊""")
    st.sidebar.title('Sidebar')
    st.sidebar.subheader("Options")
if __name__ == '__main__':
    main()


if st.sidebar.checkbox("Display Data", False):
    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")
    st.subheader("Longitude & Latitude Dataset")
    st.dataframe(lon_lat, use_container_width=st.session_state.use_container_width)
    st.write("-"*75)
    st.subheader("Marine Species Richness Dataset")
    st.dataframe(marine, use_container_width=st.session_state.use_container_width)
    st.write("-" * 75)
    st.subheader("City Temperature")
    st.dataframe(temp.head(), use_container_width=st.session_state.use_container_width)
    st.write("-" * 75)

# __________________________
##Data Cleaning/ Preparation
#________________________________


lon_lat = lon_lat.drop(lon_lat.columns[[0]], axis=1)
marine = marine.drop(['C-Square Code'], axis=1)


df_info = ['Descriptive Analysis', "Shape & Columns of Dataset", "Info About Datasets", "Check NA Values",
           "Datatypes of Columns", "Visualize Marine Species"]


st.sidebar.subheader("Display Visualizations/ Analysis")
sdbar = st.sidebar.multiselect("Select:", df_info)

if 'Descriptive Analysis' in sdbar:
    st.header("Detailed Description")


    tab1, tab2, tab3 = st.tabs(["Longitude & Latitude", "Marine Species Richness", "City Temperature"])

    tab1.subheader("Description of Longitude & Latitude Datasets")
    tab1.write(lon_lat.describe())

    tab2.subheader("Description of Marine Species Dataset")
    tab2.write(marine.describe())

    tab3.subheader("Description of City Temperature Dataset")
    tab3.write(temp.describe())


if "Shape & Columns of Dataset" in sdbar:
    # Shape of the Dataset
    st.subheader("Shape of The Datasets: ")
    st.write("Shape of Longitude & Latitude Dataset   : ", lon_lat.shape)
    st.write("Shape of Marine Species Richness Dataset: ", marine.shape)
    st.write("Shape of City Temperature Dataset       : ", temp.shape)
    st.write('_'*75)

    # Columns of the Dataset
    st.subheader("Columns of the Dataset: ")
    st.write("Longitude & Latitude Dataset", lon_lat.columns)
    st.write("Marine Species Richness Dataset", marine.columns)
    st.write("City Temperature Dataset", temp.columns)
    st.write('-'*75)

if "Info About Datasets" in sdbar:
    # Information about the dataset
    st.subheader("Information About The Datasets")
    st.write(lon_lat.info())
    st.write(marine.info())
    st.write(temp.info())
    print("-" * 50)

if 'Check NA Values' in sdbar:
    st.subheader("Checking NA/ Null Values")
    ## Checking Null Values
    st.write(lon_lat.isnull().sum())
    st.write(marine.isnull().sum())
    st.write(temp.isnull().sum())

## Handling NAN Values filling missing values using fillna()
lon_lat = lon_lat.fillna(0)
marine = marine.fillna(0)
temp = temp.fillna(0)

#Creating DATE column using Month, Day & Year Columns of the Dataframe
@st.cache(allow_output_mutation=True)
def clean_data():
    temp['Date'] = temp[temp.columns[4:7]].apply(
    lambda x: '/'.join(x.dropna().astype(str)),
    axis=1
    )
    return temp
temp = clean_data()
#While converting it into datetime
# format we got a eerror saying there is year 201
#  which exist in the dataframe
# as it is irrevant we will remove the same
rslt_df = temp[(temp['Date'] == '12/3/201')]

rslt_df = temp[(temp['Year'] == 201)]

#Dropping all those rows where year == 201
index_names = temp[ temp['Year'] == 201 ].index
temp.drop(index_names, inplace = True)


#As Day cannot be zero which means it is irrelevant
rslt_df = temp[(temp['Day'] == 0)]


#Dropping all those rows where Day == 0
index_names = temp[ temp['Day'] == 0 ].index
temp.drop(index_names, inplace = True)


#As Day cannot be zero which means it is irrelevant
rslt_df = temp[(temp['Year'] == 200)]

#Dropping all those rows where Day == 0
index_names = temp[ temp['Year'] == 200 ].index
temp.drop(index_names, inplace = True)

#Converting Date column into Datetime format
temp["Date"] = pd.to_datetime(temp.Date, format="%m/%d/%Y")

if "Datatypes of Columns" in sdbar:
    st.subheader("Datatypes of Columns: ")
    st.write(lon_lat.dtypes)
    st.write(marine.dtypes)
    st.write(temp.dtypes)


def viz_m_s():
    color_scale = [(0, 'orange'), (1, 'red')]
    fig = px.scatter_mapbox(marine,
                            lat="Latitude",
                            lon="Longitude",
                            hover_name="Species Count",
                            hover_data=["Species Count"],
                            color="Species Count",
                            color_continuous_scale=color_scale,
                            size="Species Count",
                            zoom=8,
                            height=800,
                            width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

if "Visualize Marine Species" in sdbar:
    st.header("Visualization")
    st.subheader('Visualizing Marine Species According to Latitude & Longitude')
    viz_m_s()

if st.sidebar.checkbox("Apply LSTM", False):
    st.subheader("LSTM Prediction Model")
    country_list = tuple(temp.Country.unique())
    selected_country = st.selectbox('Select the Country', country_list)
    inp_country = selected_country

    df = temp[(temp['Country'] == inp_country)]
    selected_city = st.selectbox('Select the City', df.City.unique())
    inp_cty = selected_city

    df = temp[(temp['City'] == inp_cty)]
    df = pd.DataFrame(df)
    import datetime
    today = datetime.date.today()
    past = datetime.date(2018, 7, 6)
    start_date = st.date_input('Start date', past)
    end_date = st.date_input('End date', today)
    if start_date < end_date:
        st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.error('Error: End date must fall after start date.')

    ### Select rows between two dates
    start_date = str(start_date)
    end_date = str(end_date)

    mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
    df = df.loc[mask]

    # Creating Dataframe That Only Contains Date & AvgTemperature Column
    df = pd.DataFrame(df[['Date', 'AvgTemperature']])
    df.index = df['Date']  # Making Date column as Index

    if st.sidebar.checkbox("Display AvgTemp Data", False):
        st.subheader("Date & AvgTemperature Dataframe")
        # Creating Dataframe That Only Contains Date & AvgTemperature Column
        df = pd.DataFrame(df[['Date', 'AvgTemperature']])
        df.index = df['Date']  # Making Date column as Index
        st.dataframe(df)
    ####### Data Preparation
    df = df.fillna(0)

    df = df.sort_index(ascending=True, axis=0)
    data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'AvgTemperature'])

    for i in range(0, len(data)):
        data["Date"][i] = df['Date'][i]
        data["AvgTemperature"][i] = df["AvgTemperature"][i]

    ## Creating function for Calculation of Splitting of Data
    len(data)
    def split(data):
        data = len(data)
        data = ((data * 70)) / 100
        data = round(data)
        return data
    sp_val = split(data)
    st.write("Length of The Data: ", sp_val)

    ########## Min-Max Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data.index = data.Date
    data.drop("Date", axis=1, inplace=True)
    final_data = data.values
    train_data = final_data[0:sp_val, :]
    valid_data = final_data[sp_val:, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_data)
    x_train_data, y_train_data = [], []
    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60:i, 0])
        y_train_data.append(scaled_data[i, 0])


    ########LSTM Model

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(np.shape(x_train_data)[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    model_data = data[len(data) - len(valid_data) - 60:].values
    model_data = model_data.reshape(-1, 1)
    model_data = scaler.transform(model_data)

    #######Training and Testing Data
    ##Converting the the values into arrays to avoid errors
    # df = df.fillna(0)
    x_train_data = np.asarray(x_train_data)
    y_train_data = np.asarray(y_train_data)

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

    X_test = []
    for i in range(60, model_data.shape[0]):
        X_test.append(model_data[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    ####### Model Evaluation
    model_eval = lstm_model.summary()


    ##### Prediction Function
    predicted_avg_temp = lstm_model.predict(X_test)
    predicted_avg_temp = scaler.inverse_transform(predicted_avg_temp)


    ###### Prediction Result

    df_info1 = ['Historical Average Temperature', 'Predicted Average Temperature', 'Prediction Dataframe']

    st.sidebar.subheader("Display Visualizations")
    sdbar1 = st.sidebar.multiselect("Select:", df_info1)
    
    def pred_avg_temp():
        global valid_data
        train_data = data[:sp_val]
        valid_data = data[sp_val:]

        valid_data['Predictions'] = predicted_avg_temp

        ## Plotting Using Matplotlib
        st.header("Visualization")
        st.subheader('Visualizing Predicted Average Temperature Using Model')
        plt.figure(figsize=(16, 6))
        plt.title('Prediction Of Avg Temperature Using Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Average Temperature', fontsize=18)
        plt.plot(train_data['AvgTemperature'])
        plt.plot(valid_data[['AvgTemperature', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        fig = plt.tight_layout()
        st.pyplot(fig)
        
    if 'Predicted Average Temperature' in sdbar1:
        pred_avg_temp()
        
    def hist_avg_temp():
        st.header("Visualization")
        st.subheader('Visualizing Historical View of Average Temperature')
        plt.figure(figsize=(16, 6))
        plt.title('Historical View of Average Temperature')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Average Temperature', fontsize=18)
        plt.plot(df["AvgTemperature"], label='Average Temperature history')
        fig = plt.tight_layout()
        st.pyplot(fig)
        
    if 'Historical Average Temperature' in sdbar1:
        hist_avg_temp()

    if 'Prediction Dataframe' in sdbar1:
        st.subheader("Predicted Average Temperature Dataframe")
        st.dataframe(valid_data)
st.sidebar.subheader("""I hope You Liked the Application & UI
This was made as part of a
Data Analytics project.
                        
Owned By:  Gulafshan""")

