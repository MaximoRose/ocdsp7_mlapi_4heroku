import pandas as pd
from xgboost import DMatrix
import os

def transform_X(X, loaded_model):
    """
    Applique les transformation qui precede l'estimator dans le Pipeline. Permet d'obtenir une version des donnees telle que l'estimator traitera
    """
    imputer = loaded_model['imputer']
    scaler = loaded_model['scaler']
    X_imp = imputer.transform(X)
    X_sc = scaler.transform(X_imp)
    new_X=pd.DataFrame(X_sc, columns=X.columns.tolist())
    return  new_X


def get_shap_force_xgb(df_line, loaded_model):
    """
    Retourne les coefficient de force de SHAP via l'utilisation des fonctions dédiées d'XGBoost
    """
    X_t = transform_X(df_line, loaded_model=loaded_model)
    booster = loaded_model['estimator'].get_booster()
    predictions = booster.predict(DMatrix(X_t), pred_contribs=True)
    df_pred_shap = pd.DataFrame(predictions)
    no_bias = df_pred_shap.drop(columns=[df_line.shape[1]])
    no_bias.columns = df_line.columns
    return no_bias