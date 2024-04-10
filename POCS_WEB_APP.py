import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import zipfile
import joblib

# Function to extract and load the trained image detection model
def load_image_model(zip_file_path):
    try:
        # Extract the model from the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("temp_image_model")

        # Load the model
        model = load_model("temp_image_model/pcos_detection_model.keras")

        return model

    except Exception as e:
        st.error(f"Error loading image detection model: {e}")
        return None

# Load the trained image detection model
image_model = load_image_model("pcos_detection_model.zip")

# Function to load the saved clinical data detection model
def load_clinical_model(model_path):
    try:
        model = joblib.load(open(model_path, 'rb'))
        return model

    except Exception as e:
        st.error(f"Error loading clinical data detection model: {e}")
        return None

# Load the saved clinical data detection model
clinical_model_path = 'POCS.sav'
clinical_model = load_clinical_model(clinical_model_path)

# Function for clinical data prediction
def pcos_prediction(input_data):
    try:
        input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = clinical_model.predict(input_data_reshaped)
        return 'The patient is not infected with PCOS' if prediction[0] == 0 else 'The patient is infected with PCOS'

    except Exception as e:
        st.error(f"Error predicting clinical data: {e}")
        return None

# Function to predict image
def predict_image(path):
    try:
        img = Image.open(path)
        img = img.resize((224, 224))  # Resize image to match model input size
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255
        input_arr = np.array([img_array])
        pred = image_model.predict(input_arr)
        return "Affected" if pred[0][0] < 0.5 else "Not Affected"

    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None

def main():
    st.title('PCOS Detection Web App')
    with st.sidebar:
        selected = option_menu('Polycystic Ovary Syndrome Prediction System',

                               ['Clinical Data Detection',
                                'Image Detection'],
                               menu_icon='hospital-fill',
                               icons=['activity', 'person'],
                               default_index=0)


    if selected == "Clinical Data Detection":
        st.header("Clinical Data Detection")
        image = Image.open('pocs image.png')
        st.image(image)
        # Add input fields for clinical data
        age = st.number_input('Enter Age in years:(12-50)',12,50)
        bmi = st.number_input('Enter BMI Value:(0.0-81.0)',0.0,81.0)
        bloodgroup = st.number_input('Enter Blood Group Value:(11-17)',11,17)
        pulserate = st.number_input('Enter Pulse rate in bpm:(60-100)',60,100)
        hb = st.number_input('Enter Hb in g/dl:(8.0-15.1)',8.0,15.1)
        cycle = st.number_input('Enter Cycle in R/I:(2-5)',2,5)
        cyclelength = st.number_input('Enter Cycle length in days:(2-20)',2,20)
        pregnant = st.number_input('Enter Pregnant value :(0-1)',0,1)
        ibetahcg = st.number_input('Enter I Beta-HCG value:(1.45-9649.0)',1.45,9649.0)
        fshlh = st.number_input('Enter FSH/LH value:(1.0-1400.0)',1.0,1400.0)
        tsh = st.number_input('Enter TSH value:(0.0-25.0)',0.0,25.0)
        amh = st.number_input('Enter AMH value:(0.0-25.0)',0.0,25.0)
        weightgain = st.number_input('Enter Weight gain value :(0 or 1)',0,1)
        hairloss = st.number_input('Enter Hair loss value :(0 or 1)',0,1)
        regexercise = st.number_input('Enter Regular Exercise value :(0 or 1)',0,1)
        follicleleft = st.number_input('Enter Follicle Left value:(1-25)',1,25)
        follicleright = st.number_input('Enter Follicle Right value:(1-25)',1,25)
        endometrium = st.number_input('Enter Endometrium value in mm:(0.0-15.0)',0.0,15.0)
        iibetahcg = st.number_input('Enter II Beta-HCG value:(1.45-9649)',1.45,9649.0)

        if st.button('PCOS Test Result'):
            if clinical_model:
                diagnosis = pcos_prediction([age, bmi, bloodgroup, pulserate, hb, cycle, cyclelength, pregnant, ibetahcg, fshlh, tsh, amh, weightgain, hairloss, regexercise, follicleleft, follicleright, endometrium, iibetahcg])
                if diagnosis:
                    st.success(diagnosis)
            else:
                st.error("Error: Clinical model not loaded.")

    elif selected == "Image Detection":
        st.header("Image Detection")
        image = Image.open('pocs image.png')
        st.image(image)
        uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Make prediction on the uploaded image
            if image_model:
                prediction = predict_image(uploaded_file)
                if prediction:
                    st.markdown(f"<h3>Prediction: {prediction}</h3>", unsafe_allow_html=True)
            else:
                st.error("Error: Image detection model not loaded.")

if __name__ == "__main__":
    main()
