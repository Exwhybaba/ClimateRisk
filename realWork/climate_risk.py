#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sb
import numpy as np
from datetime import datetime
from windrose import WindroseAxes
from matplotlib.cm import get_cmap
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
import folium
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, Normalizer, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,f1_score
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('POWER_Point_Monthly_Timeseries_1990_2022_011d7022N_004d0414E_LST.csv', sep= ',')


# In[3]:


df


# In[4]:


df2 = pd.read_csv('climate2.csv')
df2


# In[5]:


df2.columns


# In[6]:


melted_df = pd.melt(df2, id_vars=['PARAMETER', 'YEAR'], var_name='MONTH', value_name='VALUE')

melted_df = melted_df[melted_df['MONTH'].apply(lambda x:x != 'ANN')]
melted_df['DATE'] = pd.to_datetime(melted_df['YEAR'].astype(str) + '-' + melted_df['MONTH'], format='%Y-%b')

# Sort the DataFrame based on the 'DATE' column
sorted_df = melted_df.sort_values(by='DATE')
sorted_df


# In[7]:


sorted_df['PARAMETER'].unique()


# In[8]:


parameter_maps = {} 

for param in sorted_df['PARAMETER'].unique():
    parameter_maps[param + '_map'] = sorted_df[sorted_df['PARAMETER'] == param]

# Concatenate all DataFrames vertically
all_df = pd.concat([df.reset_index(drop=True) for df in parameter_maps.values()], axis=1)

# Print the concatenated DataFrame
all_df


# In[9]:


all_df.columns


# In[10]:


#all_df.to_csv('adjust_csv.csv')


# In[11]:


dfx = pd.read_csv('adjust_csv.csv')
dfx


# In[12]:


dfx.columns


# In[13]:


column2Drop = ['Unnamed: 0','PARAMETER', 'PARAMETER.1', 'YEAR.1', 'MONTH.1',  'DATE.1', 'PARAMETER.2', 'YEAR.2',
       'MONTH.2', 'DATE.2', 'PARAMETER.3','YEAR.3', 'MONTH.3','DATE.3','PARAMETER.4', 'YEAR.4', 'MONTH.4','DATE.4', 
        'PARAMETER.5', 'YEAR.5', 'MONTH.5', 'DATE.5', 'PARAMETER.6', 'YEAR.6','MONTH.6', 'DATE.6','PARAMETER.7',
        'YEAR.7', 'MONTH.7', 'DATE.7', 'PARAMETER.8', 'YEAR.8', 'MONTH.8', 'DATE.8']


# In[14]:


dfx.drop(columns= column2Drop, inplace= True)
dfx


# In[15]:


dfx.columns


# In[16]:


dfx = dfx[['DATE', 'YEAR', 'MONTH', 'Dew/Frost Point at 2 Meters (C)', 'Precipitation Corrected (mm/day)', 
           ' Precipitation Corrected Sum (mm)',
           'Surface Pressure (kPa)', 'Wind Speed at 2 Meters (m/s)',
       'Relative Humidity at 2 Meters (%)',
       'Specific Humidity at 2 Meters (g/kg)', 'Temperature at 2 Meters (C)',
       'Wind Speed at 10 Meters (m/s)' ]]
dfx


# In[17]:


dfx.describe().T


# In[18]:


dfx.columns


# In[19]:


dfx.info()


# In[20]:


dfn = dfx.copy()


# In[21]:


dfn['DATE'] = pd.to_datetime(dfn['DATE'])
dfn['YEAR'] = dfn['DATE'].dt.year
dfn['MONTH'] = dfn['DATE'].dt.month


# In[22]:


dfn.set_index('DATE', inplace=True)


# In[23]:


dfn


# ## Temperature Analysis

# In[24]:


fig = px.line(dfn, x= dfn.index, y='Temperature at 2 Meters (C)', 
              labels={'Temperature at 2 Meters (C)': 'Temperature (Celsius)'}, 
              title='Temperature Trends')
fig.update_traces(mode='markers+lines', marker=dict(size=8))
fig.update_layout(width=1000, height=600)
fig.show()


# In[25]:


fig = px.bar(dfn, x='MONTH', y='Temperature at 2 Meters (C)',
             labels={'Temperature at 2 Meters (C)': 'Temperature (Celsius)', 'MONTH': 'Month'},
             title='Monthly Temperature Trends')
#fig.update_traces(marker=dict(size=8))
fig.update_layout(width=1000, height=600)

fig.show()


# ## Precipitation Analysis 

# In[26]:


fig = px.line(dfn, x= dfn.index, y=' Precipitation Corrected Sum (mm)', 
              labels={' Precipitation Corrected Sum (mm)':'Precipitation(mm)'}, 
              title='Precipitation Trends')
fig.update_traces(mode='markers+lines', marker=dict(size=8))
fig.update_layout(width=1000, height=600)
fig.show()


# In[27]:


plt.figure(figsize= (11,6))
plt.style.use('seaborn-darkgrid')
color = sb.color_palette()[0]
sb.barplot(dfn, x='MONTH', y=' Precipitation Corrected Sum (mm)', color= color, errorbar=None);
plt.title('Average Monthly Precipitation', size = 16)
plt.xlabel('Month', size = 14)
plt.ylabel('Precipitation(mm)', size = 14);


# In[28]:


plt.figure(figsize= (11,6))
plt.style.use('seaborn-darkgrid')
color = sb.color_palette()[0]
sb.barplot(dfn, x='MONTH', y='Precipitation Corrected (mm/day)', color= color, errorbar=None);
plt.title('Average Monthly Rainfall Intensity', size = 16)
plt.xlabel('Month', size = 14)
plt.ylabel('Rainfall Intensity(mm/day)', size = 14);


# ## Dew/Frost Point 

# In[29]:


def plotFunc(df, x, y):
    # Plotly Express Line Plot
    fig = px.line(df, x=dfn.index, y=y,
                  labels={y: f'{y}', x: f'{x}'},
                  title=f'{y} Trends')
    fig.update_traces(mode='markers+lines', marker=dict(size=8))
    fig.update_layout(width=1000, height=600)
    fig.show()

    # Seaborn Bar Plot
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-darkgrid')
    color = sb.color_palette()[0]
    sb.barplot(data=df, x=x, y=y, color=color, errorbar=None)
    plt.title(f'Average {y}', size=16)
    plt.xlabel(x, size=14)
    plt.ylabel(f'{y}', size=14)
   


# In[30]:


plotFunc(dfn, x = 'MONTH', y = 'Dew/Frost Point at 2 Meters (C)')


# In[31]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}\n')


# In[32]:


columns = ['Surface Pressure (kPa)', 'Wind Speed at 2 Meters (m/s)','Relative Humidity at 2 Meters (%)',
       'Specific Humidity at 2 Meters (g/kg)', 'Wind Speed at 10 Meters (m/s)']

for column in columns:
    plotFunc(dfn, x = 'MONTH', y = column)


# In[ ]:





# In[33]:


correlation_matrix = dfn.corr()
plt.figure(figsize=(8, 6))
sb.heatmap(correlation_matrix, annot= True, cmap="YlGnBu", vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation Matrix', size = 16);


# ## Wind Pattern

# In[34]:


fig_time_series = px.line(dfn, x= dfn.index, y=['Wind Speed at 2 Meters (m/s)', 'Wind Speed at 10 Meters (m/s)'],
                           labels={'value': 'Wind Speed (m/s)', 'variable': 'Wind Speed at'},
                           title='Wind Speed Trends over Time')
fig_time_series.update_layout(width=1100, height=600)
fig_time_series.show()

# Wind Rose Plot for prevailing wind directions
fig_wind_rose, ax = plt.subplots(subplot_kw={'projection': 'windrose'})
cmap = get_cmap('viridis')
ax.bar(dfn['Wind Speed at 2 Meters (m/s)'], dfn['Wind Speed at 10 Meters (m/s)'], normed=True, opening=0.8, 
       edgecolor='white', color='blue', cmap = cmap)
ax.set_facecolor('lightgrey')
ax.set_title('Wind Rose - Prevailing Wind Directions')
plt.show()


# - The wind rose above shows that during this particular sampling period the wind blew from the North 100% of the time

# ## Extreme Temperature

# In[35]:


dfx[dfx['Temperature at 2 Meters (C)'].apply(lambda x : x < 22)]


# In[36]:


cold = dfn[dfn['Temperature at 2 Meters (C)'].apply(lambda x : x < 22)]
hot = dfn[dfn['Temperature at 2 Meters (C)'].apply(lambda x : x > 30)]

plt.figure(figsize=(12, 6))
plt.plot(dfn.index, dfn['Temperature at 2 Meters (C)'])
plt.scatter(x = cold.index, y = cold['Temperature at 2 Meters (C)'], c = 'b', label='Cold Snaps')
plt.scatter(x = hot.index, y = hot['Temperature at 2 Meters (C)'], c = 'r', label='Heatwaves')
plt.title('Extreme Temperature Events')
plt.xlabel('Date')
plt.ylabel('Temperature (Celsius)')
plt.legend()
plt.show()


# In[37]:


cold = dfn[dfn['Temperature at 2 Meters (C)'] < 22]
hot = dfn[dfn['Temperature at 2 Meters (C)'] > 30]

fig = go.Figure()

fig.add_trace(go.Scatter(x=dfn.index, y=dfn['Temperature at 2 Meters (C)'], 
                         mode='lines', name='Temperature'))
fig.add_trace(go.Scatter(x=cold.index, y=cold['Temperature at 2 Meters (C)'], 
                         mode='markers', marker=dict(color='blue'), 
                         name='Cold Snaps', text='Cold Snaps'))
fig.add_trace(go.Scatter(x=hot.index, y=hot['Temperature at 2 Meters (C)'],
                         mode='markers', marker=dict(color='red'), name='Heatwaves', text='Heatwaves'))

fig.update_layout(
    title='Extreme Temperature Events',
    xaxis_title='Date',
    yaxis_title='Temperature (Celsius)',
    legend_title='Event Type',
    height=600,
    width=1000
)

fig.show()


# ## Anomalies

# In[38]:


# Analyze Temperature Anomalies
mean_temperature = dfn['Temperature at 2 Meters (C)'].mean()
temperature_anomalies = dfn['Temperature at 2 Meters (C)'] - mean_temperature

# Visualize Temperature Anomalies
plt.figure(figsize=(12, 6))
plt.plot(dfn.index, temperature_anomalies, label='Temperature Anomalies', marker='o', color='red')
plt.axhline(0, color='black', linestyle='--', label='Mean Temperature')
plt.title('Temperature Anomalies over Time')
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly (Celsius)')
plt.legend()
plt.show()


# In[39]:


dfn.columns


# In[40]:


# Analyze Humidity Anomalies
mean_Relative_Humidity = dfn['Relative Humidity at 2 Meters (%)'].mean()
Relative_Humidity_anomalies = dfn['Relative Humidity at 2 Meters (%)'] - mean_Relative_Humidity

# Visualize Temperature Anomalies
plt.figure(figsize=(12, 8))
plt.plot(dfn.index, Relative_Humidity_anomalies, label='Relative_Humidity Anomalies', marker='o', color='red')
plt.axhline(0, color='black', linestyle='--', label='Mean Relative_Humidity')
plt.title('Relative_Humidity Anomalies over Time')
plt.xlabel('Date')
plt.ylabel('Relative_Humidity Anomaly (%)')
plt.legend()
plt.show()


# In[41]:


# Analyze Humidity Anomalies
mean_Specific_Humidity = dfn['Specific Humidity at 2 Meters (g/kg)'].mean()
Specific_Humidity_anomalies = dfn['Specific Humidity at 2 Meters (g/kg)'] - mean_Specific_Humidity

# Visualize Humidity Anomalies
plt.figure(figsize=(12, 6))
plt.plot(dfn.index, Specific_Humidity_anomalies, label='Specific_Humidity Anomalies', marker='o', color='red')
plt.axhline(0, color='black', linestyle='--', label='Mean Specific_Humidity')
plt.title('Specific_Humidity Anomalies over Time')
plt.xlabel('Date')
plt.ylabel('Specific_Humidity Anomaly (%)')
plt.legend()
plt.show()


# In[42]:


# Analyze Precipitation Anomalies
mean_Precipitation = dfn[' Precipitation Corrected Sum (mm)'].mean()
Precipitation_anomalies = dfn[' Precipitation Corrected Sum (mm)'] - mean_Precipitation

# Visualize Humidity Anomalies
plt.figure(figsize=(12, 6))
plt.plot(dfn.index, Specific_Humidity_anomalies, label='Precipitation Anomalies', marker='o', color='red')
plt.axhline(0, color='black', linestyle='--', label='Mean Precipitation')
plt.title('Precipitation Anomalies over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation Anomaly (mm)')
plt.legend()
plt.show()


# In[43]:


# Analyze Rain_Intensity Anomalies
mean_Rain_Intensity = dfn['Precipitation Corrected (mm/day)'].mean()
Rain_Intensity_anomalies = dfn['Precipitation Corrected (mm/day)'] - mean_Rain_Intensity

# Visualize Humidity Anomalies
plt.figure(figsize=(12, 6))
plt.plot(dfn.index, Specific_Humidity_anomalies, label='Rain_Intensity Anomalies', marker='o', color='red')
plt.axhline(0, color='black', linestyle='--', label='Mean Rain_Intensity')
plt.title('Rain_Intensity Anomalies over Time')
plt.xlabel('Date')
plt.ylabel('Rain_Intensity Anomaly (mm/day)')
plt.legend()
plt.show()


# ## Climate Trends over Year

# In[44]:


# Group data by year to analyze climate trends
dfx['DATE'] = pd.to_datetime(dfx['DATE'])
dfx['YEAR'] = dfx['DATE'].dt.year
dfi = dfx.drop(columns = ['MONTH'])
grouped_by_year = dfi.groupby('YEAR').mean()

# Visualize Temperature Trends
plt.figure(figsize=(12, 6))
plt.plot(grouped_by_year.index, grouped_by_year['Temperature at 2 Meters (C)'], marker='o', label='Mean Temperature')
plt.title('Climate Trends - Temperature over Time')
plt.xlabel('Year')
plt.ylabel('Mean Temperature (Celsius)')
plt.legend()
plt.show()


# In[45]:


plt.figure(figsize=(12, 6))
plt.plot(grouped_by_year.index, grouped_by_year['Relative Humidity at 2 Meters (%)'], marker='o', 
         label='Mean Relative Humidity')
plt.title('Climate Trends - Relative Humidity over Time')
plt.xlabel('Year')
plt.ylabel('Mean Relative Humidity (%)')
plt.legend()
plt.show()


# In[46]:


plt.figure(figsize=(12, 6))
plt.plot(grouped_by_year.index, grouped_by_year['Precipitation Corrected (mm/day)'], marker='o', 
         label='Mean Rainfall Intensity')
plt.title('Climate Trends - Rainfall Intensity over Time')
plt.xlabel('Year')
plt.ylabel('Rainfall Intensity (mm/day)')
plt.legend()
plt.show()


# In[47]:


plt.figure(figsize=(12, 6))
plt.plot(grouped_by_year.index, grouped_by_year[' Precipitation Corrected Sum (mm)'], marker='o', label='Mean Precipitation')
plt.title('Climate Trends - Precipitation over Time')
plt.xlabel('Year')
plt.ylabel('Mean Precipitation (mm/day)')
plt.legend()
plt.show()


# ## Relationship between temperature and precipitation

# In[48]:


sb.scatterplot(data= dfn, x= 'Temperature at 2 Meters (C)', y =' Precipitation Corrected Sum (mm)')
print(dfn['Temperature at 2 Meters (C)'].corr(dfn[' Precipitation Corrected Sum (mm)']))
plt.title('Relationship between Precipitation and Temperature')
plt.ylabel('Precipitation')
plt.xlabel('Temperature')


# ## Flooding Condition

# In[49]:


# Plotting the relationship between Precipitation Corrected and Surface Pressure
plt.scatter(y = dfn[' Precipitation Corrected Sum (mm)'], x = dfn['Surface Pressure (kPa)'])
plt.title('Relationship between Precipitation and Surface Pressure')
plt.ylabel('Precipitation')
plt.xlabel('Surface Pressure (kPa)')


# In[50]:


threshold_precipitation =  200
high_precipitation_events = dfn[dfn[' Precipitation Corrected Sum (mm)'] > threshold_precipitation]

# Plotting instances of high precipitation
plt.scatter( y = dfn[' Precipitation Corrected Sum (mm)'], x = dfn['Surface Pressure (kPa)'], label='Normal')
plt.scatter(y = high_precipitation_events[' Precipitation Corrected Sum (mm)'], 
            x = high_precipitation_events['Surface Pressure (kPa)'],
            color='red', label='High Precipitation Events')
plt.title('Instances of High Precipitation and Surface Pressure')
plt.ylabel('Precipitation')
plt.xlabel('Surface Pressure')


# In[51]:


plt.scatter(y = dfn['Precipitation Corrected (mm/day)'], x = dfn['Surface Pressure (kPa)'])
plt.title('Relationship between Rainfall Intensty and Surface Pressure')
plt.ylabel('Rainfall Intensity')
plt.xlabel('Surface Pressure (kPa)')


# In[52]:


low_Precipitation = dfn[dfn[' Precipitation Corrected Sum (mm)'] < 50]
high_Precipitation = dfn[dfn[' Precipitation Corrected Sum (mm)'] > 200]
fig = go.Figure()
fig.add_trace(go.Scatter(x=dfn.index, y=dfn[' Precipitation Corrected Sum (mm)'], mode='lines', name='All Data'))
# Scatter plot for low relative humidity
fig.add_trace(go.Scatter(x=low_Precipitation.index, y=low_Precipitation[' Precipitation Corrected Sum (mm)'],
                         mode='markers', marker=dict(color='blue'), name='low Precipitation'))
# Scatter plot for high relative humidity
fig.add_trace(go.Scatter(x=high_Precipitation.index, y=high_Precipitation[' Precipitation Corrected Sum (mm)'],
                         mode='markers', marker=dict(color='red'), name='high Precipitation'))
fig.update_layout(
    title='Precipitation Events',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Precipitation'),
    legend=dict(x=1, y=0, traceorder='normal')
)

fig.show()


# ## Drought Condition

# In[53]:


plt.scatter(y = dfn['Relative Humidity at 2 Meters (%)'], x = dfn['Temperature at 2 Meters (C)'])
plt.title('Relationship between Relative Humidity and Temperature')
plt.ylabel('Relative Humidity')
plt.xlabel('Temperature');


# In[54]:


dfn['Relative Humidity at 2 Meters (%)'].corr(dfn['Temperature at 2 Meters (C)'])


# In[55]:


low_relative = dfn[dfn['Relative Humidity at 2 Meters (%)'] < 30]
high_relative = dfn[dfn['Relative Humidity at 2 Meters (%)'] > 60]
fig = go.Figure()
fig.add_trace(go.Scatter(x=dfn.index, y=dfn['Relative Humidity at 2 Meters (%)'], mode='lines', name='All Data'))
# Scatter plot for low relative humidity
fig.add_trace(go.Scatter(x=low_relative.index, y=low_relative['Relative Humidity at 2 Meters (%)'],
                         mode='markers', marker=dict(color='blue'), name='Low Relative Humidity'))
# Scatter plot for high relative humidity
fig.add_trace(go.Scatter(x=high_relative.index, y=high_relative['Relative Humidity at 2 Meters (%)'],
                         mode='markers', marker=dict(color='red'), name='High Relative Humidity'))
fig.update_layout(
    title='Relative Humidity Events',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Relative Humidity'),
    legend=dict(x=1, y=0, traceorder='normal')
)

fig.show()


# In[56]:


plt.figure(figsize= (11,6))
plt.style.use('seaborn-darkgrid')
color = sb.color_palette()[0]
sb.barplot(dfn, x='MONTH', y= 'Relative Humidity at 2 Meters (%)', color= color, errorbar=None);
plt.title('Average Monthly Relative Humidity', size = 16)
plt.xlabel('Month', size = 14)
plt.ylabel('Relative Humidity(%)', size = 14);


# In[57]:


dfn['Precipitation Corrected (mm/day)'].min()


# High rainfall intensity: 5 mm/day or more
# Low rainfall intensity: 3 mm/day or less  

# In[58]:


''


# In[59]:


#mapping flood condition
flood = (dfn['Precipitation Corrected (mm/day)'] >= 5) & \
                  (dfn[' Precipitation Corrected Sum (mm)'] >= 200) & \
                  (dfn['Relative Humidity at 2 Meters (%)'] >= 60)

# mapping drought condition
drought = (dfn['Precipitation Corrected (mm/day)'] <= 3) & \
                    (dfn[' Precipitation Corrected Sum (mm)'] <= 50) & \
                    (dfn['Relative Humidity at 2 Meters (%)'] <= 30)


# In[60]:


dfn['class'] = 'normal'
dfn.loc[flood, 'class'] = 'flood'
dfn.loc[drought, 'class'] = 'drought'
dfn


# In[61]:


dfn.groupby('class').size()


# 
# 
# ## Predicting flood Model

# In[62]:


dfn.describe().T


# In[63]:


plt.figure(figsize=(25, 20), dpi= 100)
dfn.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)
plt.gcf().set_size_inches(20,20)
plt.tight_layout()
plt.show();


# In[64]:


plt.figure(figsize=(25, 20), dpi= 100)
dfn.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
plt.gcf().set_size_inches(20,20)
plt.tight_layout()
plt.show();


# In[65]:


features = [f for f in dfn.columns if f != 'class']
target = [t for t in dfn.columns if t == 'class']



# In[66]:


dfn.groupby('class').size()


# In[67]:


feature_df = dfn[features]
label_df = dfn[target]


# In[68]:


dfn.columns.get_loc('class') 


# In[69]:


oversample = SMOTENC(sampling_strategy='auto', categorical_features=[0])
tfrm_features, tfrm_target = oversample.fit_resample(feature_df, label_df)


# In[70]:


print(f'new label count: {tfrm_target.value_counts()}')
print(f'old label count: {label_df.value_counts()}')


# In[71]:


dfi = pd.concat([tfrm_features,tfrm_target],axis=1, join='outer')
dfi


# In[72]:


dfi.groupby('class').size()


# In[118]:


dfi.columns


# In[73]:


X = dfi[features].values
y = dfi[target].values


# ## Preprocessing

# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=7)


# In[75]:


#Encoding categorical data
encoder = LabelEncoder().fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


# In[76]:


X_train_df = pd.DataFrame(X_train, columns = features)
X_train_df.describe().T


# In[77]:


X_test_df = pd.DataFrame(X_test, columns = features)
y_train_df = pd.DataFrame(y_train, columns = target)
y_test_df = pd.DataFrame(y_test, columns = target)


# In[78]:


y_train_df.value_counts()


# In[79]:


y_test_df.value_counts()


# In[80]:


#Rescaling
scaler = MinMaxScaler().fit(X_train)
RX_train = scaler.transform(X_train)
RX_test = scaler.transform(X_test)


# In[81]:


RX_train_df = pd.DataFrame(RX_train, columns = features)
RX_train_df.describe().T


# In[82]:


#Normalization
scaler = Normalizer().fit(RX_train)
NRX_train = scaler.transform(RX_train)
NRX_test = scaler.transform(RX_test)


# In[83]:


NRX_train_df = pd.DataFrame(NRX_train, columns = features)
NRX_train_df.describe().T


# In[84]:


NRX_test_df = pd.DataFrame(NRX_test, columns = features)
NRX_test_df.describe().T


# In[85]:


plt.figure(figsize=(25, 20), dpi= 100)
NRX_train_df.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)
plt.gcf().set_size_inches(20,20)
plt.tight_layout()
plt.show();


# ## Multicolinearity

# In[86]:


NRX_train_df['class'] = y_train


# In[87]:


corr = NRX_train_df.corr()
plt.figure(figsize=(35, 15))
sb.heatmap(corr, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, fmt=".2f", mask=np.triu(np.ones_like(corr, dtype=bool)))


# In[88]:


dfi.columns


# In[89]:


column2Drop = ['Precipitation Corrected (mm/day)', 'Specific Humidity at 2 Meters (g/kg)', 
               'Dew/Frost Point at 2 Meters (C)', 'Wind Speed at 10 Meters (m/s)','Relative Humidity at 2 Meters (%)']


# In[90]:


NRX_train_df.drop(columns = column2Drop, inplace= True)


# In[91]:


corr = NRX_train_df.corr()
plt.figure(figsize=(35, 15))
sb.heatmap(corr, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, fmt=".2f", mask=np.triu(np.ones_like(corr, dtype=bool)))


# In[92]:


NRX_train_df.drop(columns = ['class'], inplace= True)


# In[93]:


NRX_train_df.shape


# In[94]:


from sklearn.feature_selection import RFE
X = NRX_train_df.values
Y = y_train
model = LogisticRegression() 
rfe = RFE(model, n_features_to_select=2)
fit = rfe.fit(X,Y) 
print(fit.support_)
print(fit.ranking_)


# In[95]:


RFE_ = [name for name, value in zip(NRX_train_df.columns, fit.ranking_) if value == 1]
len(RFE_)


# In[96]:


RFE_


# In[97]:


X_train = NRX_train_df[RFE_].values
X_test = NRX_test_df[RFE_].values


# In[98]:


# Spot-Check Classifier Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('Ridge', RidgeClassifier()))
models.append(('Lasso', Lasso()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[99]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[100]:


model = KNeighborsClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[101]:


# Evaluate the model and print the results
print(classification_report(y_test, predictions))
print('F1-score: ', f1_score(y_test, predictions, average='micro'))


# In[102]:


cm = confusion_matrix(y_test, predictions)
cm


# In[103]:


def make_prediction(Precipitation, RelativeHumidity):
    data = {
        ' Precipitation Corrected Sum (mm)': Precipitation,
        'Relative Humidity at 2 Meters (%)': RelativeHumidity,
    }
    
    df = pd.DataFrame(data, index=[0])

    

    prediction = model.predict(df)
    
    # Inverse transform the predicted labels
    decoded_prediction = encoder.inverse_transform(prediction)
    
    return f"It is going to be : {decoded_prediction[0]}"
 


# In[104]:


make_prediction(0, 40)


# In[105]:


a_df = dfi[[' Precipitation Corrected Sum (mm)', 'Relative Humidity at 2 Meters (%)', 'class']]
a_df


# In[106]:


a_df['predict'] = encoder.inverse_transform(model.predict(
    a_df[[' Precipitation Corrected Sum (mm)', 'Relative Humidity at 2 Meters (%)']]))


# In[107]:


a_df[a_df['class'].apply(lambda x : x == 'flood')].head(60)


# In[108]:


predictions


# # Flood Prone Areas

# In[109]:


data = {
    "Settlement": ["Jawo", "Unguwar Sani", "Makera", "Maurida", "Unguwar Kayi", "U. Mijin Nana",
                   "Unguwar Gero", "Kola", "Wuro Maliki", "Birnin Kebbi", "Ambursa", "Dagere"],
    "Latitude": [12.49750, 12.50550, 12.51389, 12.51972, 12.51472, 12.52722, 12.53056, 12.44528,
                 12.44083, 12.47389, 12.51028, 12.56417],
    "Longitude": [4.092222, 4.104722, 4.119444, 4.127500, 4.152222, 4.150000, 4.179444, 4.116111,
                  4.093056, 4.210000, 4.335000, 4.414167],
    "Elevation": [199.9, 197.8, 201.7, 200.5, 201.1, 202.0, 200.5, 204.5, 199.6, 206.7, 206.7, 205.7]
}

# Create DataFrame
df2 = pd.DataFrame(data)



# In[110]:


df2


# In[111]:


map_center = [df2['Latitude'].mean(), df2['Longitude'].mean()]


# In[112]:


kebbi_map = folium.Map(location=map_center, zoom_start=10)


# In[113]:


kebbi_map


# In[114]:


for index, row in df2.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Settlement']}: {row['Elevation']} m",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(kebbi_map)


# In[115]:


#kebbi_map.save('kebbi_elevation_map.html')
kebbi_map


# In[116]:


df2['Settlement'].unique()


# In[ ]:




