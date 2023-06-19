import pandas as pd
from xgboost import DMatrix
import os

def transform_X(X, loaded_model):
    """Applique les transformation qui precede l'estimator dans le Pipeline. Permet d'obtenir une version des donnees telle que l'estimator traitera"""
    imputer = loaded_model['imputer']
    scaler = loaded_model['scaler']
    X_imp = imputer.transform(X)
    X_sc = scaler.transform(X_imp)
    new_X=pd.DataFrame(X_sc, columns=X.columns.tolist())
    return  new_X


def get_shap_force_xgb(df_line, loaded_model):
    X_t = transform_X(df_line, loaded_model=loaded_model)
    booster = loaded_model['estimator'].get_booster()
    predictions = booster.predict(DMatrix(X_t), pred_contribs=True)
    df_pred_shap = pd.DataFrame(predictions)
    no_bias = df_pred_shap.drop(columns=[df_line.shape[1]])
    no_bias.columns = df_line.columns
    return no_bias


# TRAITEMENT DES FICHIERS DES QCUTS POUR TOP FEATURES
def get_cat_for_obs(obs, colonne, qcuts_df) :
    resulting_cat = 0
    exceeded_limits = False
    current_value = obs[colonne]
    cnt = 0
    for (borne_inf, borne_sup) in zip(qcuts_df['left_value'].values.tolist(), qcuts_df['right_value'].values.tolist()) :
        if (cnt == 0) & (current_value < borne_inf) :
            resulting_cat = 0
            exceeded_limits = True
            break
        elif (cnt == (qcuts_df.shape[0]-1)) & (current_value > borne_sup) :
            resulting_cat = qcuts_df.shape[0]-1
            exceeded_limits = True
            break
        elif (current_value > borne_inf) & (current_value <= borne_sup) :
            resulting_cat = cnt
            break
        else :
            cnt += 1
    return resulting_cat, exceeded_limits


def list_files_in_folder(folder_path):
    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            files.append(file_name)
    return files



def get_radar_values(obs, path_to_qcuts_df) :
    exceeds_train = False
    output_dict = {}
    lst_files = list_files_in_folder(path_to_qcuts_df)
    for file in lst_files :
        feat_name = file.split('.')[0]
        qcut_df = pd.read_csv(path_to_qcuts_df+file)
        cat, ex_cat = get_cat_for_obs(obs, feat_name, qcut_df)
        output_dict[feat_name] = cat
        if ex_cat :
            exceeds_train=True

    if exceeds_train :
        output_dict['ExceedsKnownData'] = 1
    else :
        output_dict['ExceedsKnownData'] = 0
    return output_dict
