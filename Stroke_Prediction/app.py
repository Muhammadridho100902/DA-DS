import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('Stroke_Prediction\stroke_model.sav', 'rb'))


def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not stroke'
    else:
        return 'The person is stroke'


def main():

    # for title
    st.title('Stroke Prediction Web APP')

    # Labels
    st.write('Gender : (1 for Male), (0 for Female)')
    # getting the input data from the user
    Gender = st.select_slider("Choose Gender", [0, 1])
    Age = st.text_input('Age')

    st.write('Hypertension : 0 if the patient doesn\'t have hypertension, 1 if the patient has hypertension')
    Hypertension = st.select_slider('Hypertension', [0, 1])

    st.write('Heart Disease : 0 if the patient doesn\'t have any heart diseases, 1 if the patient has a heart disease')
    Heart_disease = st.select_slider('Heart Disease', [0, 1])

    st.write(
        'Ever Married : 0 if the patient doesn\'t have married, 1 if the patient have married')
    Ever_married = st.select_slider('Ever Married', [0, 1])
    Avg_glucose_level = st.text_input('Glukosa Level')
    Bmi = st.text_input(
        'BMI Value')
    
    st.write('Smoking Status : 0 if the patient doesn\'t smoke, 1 if the patient has smoke')
    Smoking_status = st.select_slider('Are you Smoking', [0, 1])

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Stroke Test Result'):
        diagnosis = diabetes_prediction(
            [Gender, Age, Hypertension, Heart_disease, Ever_married, Avg_glucose_level, Bmi, Smoking_status])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
