import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os

def main():

    df = pd.read_csv('data/credit_train.csv')


    X = df.drop('Y', axis=1)  
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicamos SMOTE para tratar el desbalance de las clases
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote) 
    X_test_scaled = scaler.transform(X_test)  

    # Utilizamos un modelo XGBoost
    model = XGBClassifier(random_state=42, eval_metric='logloss')

    # Hiperparámetros a optimizar
    param_grid = {
        'n_estimators': [100, 300, 500, 800],     
        'learning_rate': [0.1, 0.05],  
        'max_depth': [5, 7, 9]        
    }

    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    grid_search.fit(X_train_scaled, y_train_smote)

    best_params = grid_search.best_params_

    model = grid_search.best_estimator_

    y_pred = model.predict(X_test_scaled)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Best Hyperparameters from GridSearchCV:")
    print(best_params)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    if not os.path.exists('model'):
        os.makedirs('model')

    with open("model/xgboost_model.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("model/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

if __name__ == '__main__':
    main()
