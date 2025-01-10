import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

lb = LabelEncoder()
sc = StandardScaler()

DATASET_PATH = './archive/voice.csv'
MODEL_OUTPUT_PATH = './model.pkl'

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")
    print(f"file path: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"DataFrame: {df}")
    if 'label' not in df.columns:
        raise ValueError("No dataset 'label' col.")
    
    X = df.drop(columns=['label']).values
    y = df['label'].values
    print(f"X: {X}, \nLength X: {len(X)}")
    print(f"y: {y}, \nLength y: {len(y)}")

    print(f"Feature engineering")
    y = lb.fit_transform(y)

    return X, y

def train_model(X, y, model_output_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    model = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

def main():
    try:
        print("Loading dataset...")
        X, y = load_dataset(DATASET_PATH)
        
        print("Training the model...")
        train_model(X, y, MODEL_OUTPUT_PATH)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
