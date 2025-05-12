from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle (assurez-vous que le modèle 'model.joblib' est dans le bon dossier)
model = joblib.load('knn_DIABET.pkl')

# Route de la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour effectuer la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Récupération des valeurs du formulaire
    age = int(data['age'])
    pregnancies = int(data['pregnancies'])
    bmi = float(data['bmi'])
    glucose = float(data['glucose'])
    blood_pressure = float(data['blood_pressure'])
    hba1c = float(data['hba1c'])
    ldl = float(data['ldl'])
    hdl = float(data['hdl'])
    triglycerides = float(data['triglycerides'])
    waist_circumference = float(data['waist_circumference'])
    hip_circumference = float(data['hip_circumference'])
    whr = float(data['whr'])
    family_history = int(data['family_history'])
    diet_type = int(data['diet_type'])
    hypertension = int(data['hypertension'])
    medication_use = int(data['medication_use'])
    
    # Création du vecteur de features pour la prédiction
    features = np.array([[age, pregnancies, bmi, glucose, blood_pressure, hba1c, 
                          ldl, hdl, triglycerides, waist_circumference, hip_circumference, 
                          whr, family_history, diet_type, hypertension, medication_use]])

    # Prédiction
    prediction = model.predict(features)[0]
    result = "Diabète détecté" if prediction == 1 else "Pas de diabète détecté"
    
    return render_template('index.html', prediction=result)

# Démarrage de l'application
if __name__ == '__main__':
    app.run(debug=True)
