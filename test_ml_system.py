import pytest
import pandas as pd
import numpy as np
import time
import joblib
import os
from my_model import train_and_save

# Chargement initial pour les tests de succès
model, X_test, y_test = train_and_save()

# =================================================================
# 1. TESTS DE DONNÉES (Data Quality) [cite: 243]
# =================================================================

def test_data_schema_success():
    """SUCCÈS : Vérifie que le schéma est correct [cite: 248]"""
    df = pd.read_csv("data/sample_input.csv")
    expected_columns = ["age", "revenue", "history"]
    assert all(col in df.columns for col in expected_columns)

def test_data_schema_fail():
    """ÉCHEC : Simule un schéma corrompu (colonne manquante)"""
    bad_df = pd.DataFrame({"age": [25], "revenue": [50000]}) # 'history' manque
    expected_columns = ["age", "revenue", "history"]
    with pytest.raises(AssertionError):
        assert all(col in bad_df.columns for col in expected_columns)

def test_no_negative_values_success():
    """SUCCÈS : L'âge doit être positif [cite: 248]"""
    df = pd.read_csv("data/sample_input.csv")
    assert (df['age'] >= 0).all()

def test_no_negative_values_fail():
    """ÉCHEC : Détecte des valeurs aberrantes (Data Drift) """
    drift_data = pd.DataFrame({"age": [-5, 20], "revenue": [10, 20], "history": [1, 2]})
    with pytest.raises(AssertionError):
        assert (drift_data['age'] >= 0).all()

# =================================================================
# 2. TESTS DE MODÈLE (Model Validity) [cite: 244]
# =================================================================

def test_model_output_shape_success():
    """SUCCÈS : Le modèle renvoie une prédiction valide [cite: 248]"""
    sample = np.array([[5.1, 3.5, 1.4]])
    prediction = model.predict(sample)
    assert len(prediction) == 1

def test_model_precision_success():
    """SUCCÈS : La performance est supérieure au seuil [cite: 245]"""
    score = model.score(X_test, y_test)
    assert score > 0.80

def test_model_precision_fail():
    """ÉCHEC : Simule une dégradation du modèle (Model Decay) [cite: 260]"""
    low_score = 0.50 # On simule un score faible
    with pytest.raises(AssertionError):
        assert low_score > 0.80

# =================================================================
# 3. TESTS DE PERFORMANCE (Infrastructure) [cite: 245]
# =================================================================

def test_prediction_latency_success():
    """SUCCÈS : Temps de réponse rapide (< 100ms) [cite: 245]"""
    sample = np.array([[5.1, 3.5, 1.4]])
    start = time.time()
    model.predict(sample)
    latency = time.time() - start
    assert latency < 0.1

def test_prediction_latency_fail():
    """ÉCHEC : Simule une surcharge système (latence trop élevée)"""
    time.sleep(0.2) # On force un délai
    latency = 0.2
    with pytest.raises(AssertionError):
        assert latency < 0.1



def test_production_latency_strict_requirement():
    """
    CE TEST VA ÉCHOUER (FAILURE).
    Simule une exigence métier trop stricte : latence < 0.0000001s.
    L'échec de ce test signalerait que l'infrastructure n'est pas assez rapide.
    """
    # Utilisation du DataFrame pour éviter le Warning
    sample = pd.DataFrame([[5.1, 3.5, 1.4]], columns=["age", "revenue", "history"])
    
    start = time.time()
    model.predict(sample)
    latency = time.time() - start
    
    # Cette assertion sera fausse, car Python ne peut pas prédire aussi vite
    assert latency < 0.0000001