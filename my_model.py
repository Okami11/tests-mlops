import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def prepare_data():
    # Collecte des données
    iris = load_iris()
    X = pd.DataFrame(iris.data[:, :3], columns=["age", "revenue", "history"]) 
    y = iris.target
    
    # Export pour illustration pratique
    os.makedirs("data", exist_ok=True)
    X.head().to_csv("data/sample_input.csv", index=False)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save():
    X_train, X_test, y_train, y_test = prepare_data()
    # Modélisation
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    # Sauvegarde du modèle (Artifact)
    joblib.dump(model, "model.joblib")
    return model, X_test, y_test

if __name__ == "__main__":
    train_and_save()