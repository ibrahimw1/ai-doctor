"""
@author: ibrahimwani
"""

import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from models_from_scratch import Logistic_Regression, SVM_classifier, Linear_Regression, SigmoidPerceptron, preprocess_image

# loading the saved models

diabetes_model = pickle.load(open('saved_models/diabetes_model.sav', 'rb'))
diabetes_scaler = pickle.load(open('saved_models/diabetes_scaler.sav', 'rb'))

diabetes_nn_model = pickle.load(open('saved_models/diabetes_nn_model.sav', 'rb'))
diabetes_nn_scaler = pickle.load(open('saved_models/diabetes_nn_scaler.sav', 'rb'))

heart_disease_model = pickle.load(open('saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('saved_models/parkinsons_model.sav', 'rb'))
parkinsons_scaler = pickle.load(open('saved_models/parkinsons_scaler.sav', 'rb'))

medical_insurance_model = pickle.load(open('saved_models/medical_insurance_model.sav', 'rb'))

breast_cancer_model = pickle.load(open('saved_models/breast_cancer_model.sav', 'rb'))
breast_cancer_scaler = pickle.load(open('saved_models/breast_cancer_scaler.sav', 'rb'))

face_mask_model = pickle.load(open('saved_models/face_mask_model.sav', 'rb'))


# sidebar for navigate
with st.sidebar:
    
    selected = option_menu('AI Doctor',
                           ['Home',
                            'Face Mask Prediction',
                            'Breast Cancer Prediction',
                            'Diabetes Prediction',
                            'Diabetes Prediction 2.0',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Medical Insurance Prediction'
                           ],
                           icons = ['house-door', 'virus', 'hospital', 'activity', 'activity','heart', 'person', 'cash'],
                           menu_icon='hospital-fill',
                           default_index = 0)

# Main Page
if selected == 'Home':
    st.title("🏥")
    st.title('AI Doctor - Transforming Healthcare with AI')
    st.write(
        "Welcome to AI Doctor! 🌟 This revolutionary application is at the forefront of transforming healthcare "
        "using the power of Artificial Intelligence. Explore the possibilities as we leverage machine learning models "
        "to provide accurate predictions for various medical conditions.\n\n"
        "💡 **Key Features:**\n"
        "   - **Face Mask Prediction:** Our Face Mask Prediction module employs a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask.\n\n"
        "   - **Breast Cancer Prediction:** Leveraging the power of a custom Neural Network for Breast Cancer Prediction, ensuring accurate predictions for the nature of breast tumors.\n"
        "   - **Diabetes Prediction:** AI Doctor employs a custom Machine Learning Support Vector Machine (SVM) Classifier from scratch to predict diabetes risk with precision.\n"
        "   - **Diabetes 2.0:** Our advanced Diabetes 2.0 module uses a custom Deep Learning Perceptron Neural Network from scratch, providing enhanced accuracy and reliability.\n"
        "   - **Heart Disease Prediction:** Utilizing a custom Machine Learning Logistic Regression model crafted from scratch to identify heart disease risks early on.\n"
        "   - **Parkinson's Prediction:** The Parkinson's Prediction module uses a Support Vector Machine (SVM) Classifier tested with GridSearchCV for optimal performance.\n"
        "   - **Medical Insurance Prediction:** The Medical Insurance Prediction model is based on a custom Linear Regression model from scratch, offering personalized insurance cost estimates based on individual factors.\n"
        "🚀 Join us on this exciting journey as we harness technology to make healthcare accessible, efficient, and innovative."
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    # Get user input for diabetes prediction
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', value=0)
        
    with col2:
        Glucose = st.number_input('Glucose Level', value=0)
        
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', value=0)
        
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', value=0)
        
    with col2:
        Insulin = st.number_input('Insulin Level', value=0)
    
    with col3:
        BMI = st.number_input('BMI value')
        
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.number_input('Age of the Person', value=0)

    # Prediction
    diab_diagnosis = ''
    
    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        # predict for one person
        individual_data_point = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Standardize the individual data point using the same scaler used for training data
        individual_data_point_scaled = diabetes_scaler.transform(individual_data_point.reshape(1, -1))
        
        # Use the trained perceptron to predict the outcome for the individual data point
        diab_prediction = diabetes_model.predict(individual_data_point_scaled)
        
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction 2.0':
    # page title
    st.title('Diabetes Prediction using Deep Learning')
    
    col1, col2, col3 = st.columns(3)
    
    # Get user input for diabetes prediction
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', value=0)
        
    with col2:
        Glucose = st.number_input('Glucose Level', value=0)
        
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', value=0)
        
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', value=0)
        
    with col2:
        Insulin = st.number_input('Insulin Level', value=0)
    
    with col3:
        BMI = st.number_input('BMI value')
        
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.number_input('Age of the Person', value=0)

    # Prediction
    diab_diagnosis = ''
    
    # creating a button for prediction
    if st.button('Diabetes  Test Result'):
        # predict for one person
        individual_data_point = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Standardize the individual data point using the same scaler used for training data
        individual_data_point_scaled = diabetes_nn_scaler.transform(individual_data_point.reshape(1, -1))
        
        # Use the trained perceptron to predict the outcome for the individual data point
        prediction = diabetes_nn_model.predict(individual_data_point_scaled)
        
        # Extract the scalar value from the NumPy array
        prediction_scalar = prediction.item()

        
        # Convert the prediction to a class (0 or 1) based on the threshold (0.5)
        predicted_class = 1 if prediction_scalar >= 0.5 else 0
        
        # Display the result
        if predicted_class == 1:
            diab_diagnosis = f"The person is diabetic with a probability of {round(prediction_scalar * 100, 2)}%"
        else:
            diab_diagnosis = f"The person is not diabetic with a probability of {round((1 - prediction_scalar) * 100, 2)}%"
        
    st.success(diab_diagnosis)
    


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', value=0)
        
    with col2:
        sex = st.number_input('Sex', value=0)
        
    with col3:
        cp = st.number_input('Chest Pain types', value=0)
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', value=0)
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', value=0)
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl', value=0)
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results', value=0)
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', value=0)
        
    with col3:
        exang = st.number_input('Exercise Induced Angina', value=0)
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', value=0)
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment', value=0)
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy', value=0)
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', value=0)
     
    # Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        # convert inputs to the appropriate data type
        inputs = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]])
        
        heart_prediction = heart_disease_model.predict(inputs)
        
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
    
    st.success(heart_diagnosis)


    
# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        input_data = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        input_data_reshaped = input_data.reshape(1,-1)
        
        std_data = parkinsons_scaler.transform(input_data_reshaped)
        
        parkinsons_prediction = parkinsons_model.predict(std_data)

        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
    
if selected == 'Medical Insurance Prediction':
    # page title
    st.title('Medical Insurance Prediction using ML')

    col1, col2, col3 = st.columns(3)

    # Get user input for medical insurance prediction
    with col1:
        age = st.number_input('Age', value=0)

    with col2:
        sex_options = ['Male', 'Female']
        sex = st.selectbox('Sex', sex_options)

    with col3:
        bmi = st.number_input('BMI')

    with col1:
        children = st.number_input('Number of Children', value=0)

    with col2:
        smoker_options = ['Yes', 'No']
        smoker = st.selectbox('Smoker', smoker_options)

    with col3:
        region_options = ['Southeast', 'Southwest', 'Northeast', 'Northwest']
        region = st.selectbox('Region', region_options)

    # Encode categorical variables
    sex_mapping = {'Male': 0, 'Female': 1}
    smoker_mapping = {'Yes': 0, 'No': 1}
    region_mapping = {'Southeast': 0, 'Southwest': 1, 'Northeast': 2, 'Northwest': 3}

    # Map user inputs to encoded values
    sex_encoded = sex_mapping.get(sex, 0)
    smoker_encoded = smoker_mapping.get(smoker, 0)
    region_encoded = region_mapping.get(region, 0)

    # Predict
    insurance_features = [age, sex_encoded, bmi, children, smoker_encoded, region_encoded]
    insurance_features = np.array(insurance_features).reshape(1, -1)  # Reshape to ensure it's a 2D array

    # Prediction
    insurance_cost = ''

    # Creating a button for prediction
    if st.button('Calculate Insurance Cost'):
        # Predict for one person
        insurance_prediction = medical_insurance_model.predict(insurance_features)

        # Handle the case where the predicted cost is less than 0
        insurance_prediction = max(0, insurance_prediction[0])

        insurance_cost = f'The estimated insurance cost is ${insurance_prediction:,.2f}'

    st.success(insurance_cost)
    
    
# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    # page title
    st.title('Breast Cancer Prediction using Deep Learning')

    # Get user input for breast cancer prediction
    col1, col2, col3 = st.columns(3)

    # Features Set 1
    with col1:
        radius_mean = st.number_input('Radius Mean', value=0.0)

    with col2:
        texture_mean = st.number_input('Texture Mean', value=0.0)

    with col3:
        perimeter_mean = st.number_input('Perimeter Mean', value=0.0)

    # Features Set 2
    with col1:
        area_mean = st.number_input('Area Mean', value=0.0)

    with col2:
        smoothness_mean = st.number_input('Smoothness Mean', value=0.0, format="%.5f")

    with col3:
        compactness_mean = st.number_input('Compactness Mean', value=0.0, format="%.5f")

    # Features Set 3
    with col1:
        concavity_mean = st.number_input('Concavity Mean', value=0.0, format="%.5f")

    with col2:
        concave_points_mean = st.number_input('Concave Points Mean', value=0.0, format="%.5f")

    with col3:
        symmetry_mean = st.number_input('Symmetry Mean', value=0.0, format="%.5f")

    # Features Set 4
    with col1:
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean', value=0.0, format="%.5f")

    with col2:
        radius_se = st.number_input('Radius SE', value=0.0, format="%.5f")

    with col3:
        texture_se = st.number_input('Texture SE', value=0.0, format="%.4f")

    # Features Set 5
    with col1:
        perimeter_se = st.number_input('Perimeter SE', value=0.0, format="%.3f")

    with col2:
        area_se = st.number_input('Area SE', value=0.0)

    with col3:
        smoothness_se = st.number_input('Smoothness SE', value=0.0, format="%.6f")

    # Features Set 6
    with col1:
        compactness_se = st.number_input('Compactness SE', value=0.0, format="%.5f")

    with col2:
        concavity_se = st.number_input('Concavity SE', value=0.0, format="%.5f")

    with col3:
        concave_points_se = st.number_input('Concave Points SE', value=0.0, format="%.5f")

    # Features Set 7
    with col1:
        symmetry_se = st.number_input('Symmetry SE', value=0.0, format="%.5f")

    with col2:
        fractal_dimension_se = st.number_input('Fractal Dimension SE', value=0.0, format="%.6f")

    with col3:
        radius_worst = st.number_input('Radius Worst', value=0.0)

    # Features Set 8
    with col1:
        texture_worst = st.number_input('Texture Worst', value=0.0)

    with col2:
        perimeter_worst = st.number_input('Perimeter Worst', value=0.0)

    with col3:
        area_worst = st.number_input('Area Worst', value=0.0)

    # Features Set 9
    with col1:
        smoothness_worst = st.number_input('Smoothness Worst', value=0.0, format="%.4f")

    with col2:
        compactness_worst = st.number_input('Compactness Worst', value=0.0, format="%.4f")

    with col3:
        concavity_worst = st.number_input('Concavity Worst', value=0.0, format="%.4f")

    # Features Set 10
    with col1:
        concave_points_worst = st.number_input('Concave Points Worst', value=0.0, format="%.5f")

    with col2:
        symmetry_worst = st.number_input('Symmetry Worst', value=0.0, format="%.4f")

    with col3:
        fractal_dimension_worst = st.number_input('Fractal Dimension Worst', value=0.0, format="%.5f")


    # Prediction
    cancer_diagnosis = ''

    # Creating a button for prediction
    if st.button('Breast Cancer Test Result'):
        # Convert user inputs to a numpy array
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                                concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
                                texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                                concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                                perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                                concave_points_worst, symmetry_worst, fractal_dimension_worst]])

        # Standardize the input data
        input_data_std = breast_cancer_scaler.transform(input_data)

        # Use the trained model to predict the outcome
        cancer_prediction = breast_cancer_model.predict(input_data_std)
        
        prediction_label = [np.argmax(cancer_prediction)]

        if prediction_label[0] == 1:
            cancer_diagnosis = 'The tumor is Malignant'
        else:
            cancer_diagnosis = 'The tumor is Benign'

    st.success(cancer_diagnosis)
    
# Face Mask Prediction
if selected == 'Face Mask Prediction':
    st.title('Face Mask Prediction')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Preprocess the image and make prediction
        img_array = preprocess_image(uploaded_file)
        prediction = face_mask_model.predict(img_array)
        
        input_pred_label = np.argmax(prediction)
        
        if input_pred_label == 1:
            st.success('The person in the image is wearing a face mask.')
        else:
            st.error('The person in the image is not wearing a face mask.')
