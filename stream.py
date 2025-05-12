import streamlit as st
import numpy as np
import joblib
import pickle

# Charger le modèle
model = joblib.load('C:\\Users\\Probook\\Desktop\\fati\\knn_DIABET.pkl')
# Titre de l'application
st.title('Prédiction Diabète')

# Collecte des données via des widgets Streamlit
age = st.number_input('Âge', min_value=1, max_value=120, value=30)
pregnancies = st.number_input('Nombre de grossesses', min_value=0, value=0)
bmi = st.number_input('IMC', min_value=0.0, max_value=100.0, value=25.0)
glucose = st.number_input('Glucose', min_value=0.0, max_value=1000.0, value=90.0)
blood_pressure = st.number_input('Pression artérielle', min_value=0.0, max_value=200.0, value=80.0)
hba1c = st.number_input('HbA1c', min_value=0.0, max_value=20.0, value=5.5)
ldl = st.number_input('LDL', min_value=0.0, max_value=300.0, value=100.0)
hdl = st.number_input('HDL', min_value=0.0, max_value=100.0, value=50.0)
triglycerides = st.number_input('Triglycérides', min_value=0.0, max_value=1000.0, value=150.0)
waist_circumference = st.number_input('Circonférence du ventre', min_value=0.0, max_value=200.0, value=80.0)
hip_circumference = st.number_input('Circonférence des hanches', min_value=0.0, max_value=200.0, value=90.0)
whr = st.number_input('WHR', min_value=0.0, max_value=1.0, value=0.9)
family_history = st.number_input('Antécédents familiaux', min_value=0, max_value=1, value=0)
diet_type = st.number_input('Type de régime', min_value=0, max_value=1, value=0)
hypertension = st.number_input('Hypertension', min_value=0, max_value=1, value=0)
medication_use = st.number_input('Utilisation de médicaments', min_value=0, max_value=1, value=0)

# Créer le vecteur de features
features = np.array([[age, pregnancies, bmi, glucose, blood_pressure, hba1c,
                      ldl, hdl, triglycerides, waist_circumference, hip_circumference, 
                      whr, family_history, diet_type, hypertension, medication_use]])

# Bouton pour effectuer la prédiction
if st.button('Faire la prédiction'):
    # Prédiction
    prediction = model.predict(features)[0]
    result = "Diabète détecté" if prediction == 1 else "Pas de diabète détecté"
    
    # Afficher le résultat
    st.write(f"### Résultat de la prédiction : {result}")

