 
# Install required libraries
!pip install pandas scikit-learn kagglehub joblib ipywidgets

import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import joblib

# Load the saved model and preprocessor
try:
    kmeans = joblib.load('/content/kmeans_model.pkl')
    preprocessor = joblib.load('/content/preprocessor.pkl')
except FileNotFoundError:
    print("Error: Model or preprocessor file not found. Run triage_clustering.py first.")
    raise

# Create input widgets
age = widgets.IntText(value=45, description='Age:', style={'description_width': 'initial'})
gender = widgets.Dropdown(options=['Male', 'Female'], value='Male', description='Gender:', style={'description_width': 'initial'})
admission_type = widgets.Dropdown(options=['Emergency', 'Elective', 'Urgent'], value='Emergency', description='Admission Type:', style={'description_width': 'initial'})
medical_condition = widgets.Dropdown(options=['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Cancer', 'Obesity'], value='Diabetes', description='Medical Condition:', style={'description_width': 'initial'})
length_of_stay = widgets.IntText(value=5, description='Length of Stay (days):', style={'description_width': 'initial'})
billing_amount = widgets.FloatText(value=25000.0, description='Billing Amount:', style={'description_width': 'initial'})

# Create button and output
button = widgets.Button(description="Predict Cluster", button_style='success')
output = widgets.Output()

# Define prediction function
def on_button_clicked(b):
    with output:
        output.clear_output()
        try:
            # Collect input data
            new_patient = pd.DataFrame({
                'Age': [age.value],
                'Gender': [gender.value],
                'Admission Type': [admission_type.value],
                'Medical Condition': [medical_condition.value],
                'Length of Stay': [length_of_stay.value],
                'Billing Amount': [billing_amount.value]
            })
            
            # Preprocess and predict
            processed_new = preprocessor.transform(new_patient)
            predicted_cluster = kmeans.predict(processed_new)[0]
            
            # Display result
            print(f"Predicted Cluster: {predicted_cluster}")
            print("(0: Low-risk, 1: Moderate, 2: Urgent)")
        except Exception as e:
            print(f"Error: {str(e)}")

button.on_click(on_button_clicked)

# Display widgets
display(age, gender, admission_type, medical_condition, length_of_stay, billing_amount, button, output)
 