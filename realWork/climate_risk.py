import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.metrics import accuracy_score

# Function to load and preprocess the data
def load_and_preprocess_data():
    path = 'https://raw.githubusercontent.com/Exwhybaba/ClimateRisk/main/realWork/adjust_csv.csv'
    dfx = pd.read_csv(path, sep=',', encoding='utf-8')

    # Your data preprocessing code here...

    return dfx

# Function to train the model
def train_model(X_train, y_train):
    # Your model training code here...
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def make_prediction(model, Precipitation, RelativeHumidity):
    data = {
        'Precipitation Corrected Sum (mm)': Precipitation,
        'Relative Humidity at 2 Meters (%)': RelativeHumidity,
    }
    df = pd.DataFrame(data, index=[0])

    prediction = model.predict(df)
    return prediction

# Main Streamlit app
def main():
    st.title("Climate Risk Prediction App")

    # Load and preprocess the data
    df = load_and_preprocess_data()

    # Display raw data
    st.subheader("Raw Data")
    st.write(df)

    # Model Training
    st.subheader("Model Training")

    # Separate features and target
    features = df[['Precipitation Corrected Sum (mm)', 'Relative Humidity at 2 Meters (%)']]
    target = df['class']

    # Oversample using SMOTE
    oversample = SMOTENC(sampling_strategy='auto', categorical_features=[0])
    tfrm_features, tfrm_target = oversample.fit_resample(features, target)

    # Encode labels
    encoder = LabelEncoder().fit(tfrm_target)
    y_train = encoder.transform(tfrm_target)

    # Rescale and Normalize
    scaler = MinMaxScaler().fit(tfrm_features)
    RX_train = scaler.transform(tfrm_features)
    scaler = Normalizer().fit(RX_train)
    NRX_train = scaler.transform(RX_train)

    # Feature selection using RFE
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=2)
    fit = rfe.fit(NRX_train, y_train)
    selected_features = [f for f, s in zip(features.columns, fit.support_) if s]

    st.write("Selected Features:", selected_features)

    # Train the model
    model = train_model(NRX_train, y_train)

    # Model Evaluation
    st.subheader("Model Evaluation")
    X_test = NRX_train  # Replace with your test data
    y_test = y_train  # Replace with your test labels

    predictions = model.predict(X_test)

    # Evaluate the model and print the results
    st.write("Classification Report:")
    st.write(classification_report(y_test, predictions))

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, predictions))

    # Streamlit Input for Prediction
    st.subheader("Make Prediction")
    st.write("Enter values for Precipitation and Relative Humidity to make a prediction:")
    precipitation_input = st.slider("Precipitation Corrected Sum (mm)", 0, 500, 50)
    humidity_input = st.slider("Relative Humidity at 2 Meters (%)", 0, 100, 50)

    if st.button("Make Prediction"):
        prediction = make_prediction(model, precipitation_input, humidity_input)
        decoded_prediction = encoder.inverse_transform([prediction])[0]
        st.write(f"The predicted class is: {decoded_prediction}")

if __name__ == "__main__":
    main()
