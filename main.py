from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

REF_FEATURES = ['AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                'DAYS_ID_PUBLISH', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
                'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                'CREDIT_TO_ANNUITY_RATIO', 'CREDIT_TO_GOODS_RATIO', 'ANNUITY_TO_INCOME_RATIO',
                'INCOME_TO_EMPLOYED_RATIO', 'INCOME_TO_BIRTH_RATIO', 'NAME_EDUCATION_TYPE_CATed',
                'BURO_DAYS_CREDIT_MEAN', 'BURO_CREDIT_ENDDATE_MAX', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
                'BURO_AMT_CREDIT_SUM_LIMIT_MEAN', 'BURO_DEBT_PERCENTAGE_MEAN', 'BURO_HAS_CREDIT_SUM',
                'BURO_HAS_CLOSED_SUM', 'BURO_HAS_DELAYED_BADDEBT_MEAN', 'PREV_CREDIT_TO_ANNUITY_RATIO_MEAN',
                'PREV_SIMPLE_INTERESTS_MEAN', 'PREV_DOWN_PAYMENT_TO_CREDIT_MEAN', 
                'PREV_DAYS_LAST_DUE_DIFF_MEAN', 'PREV_YIELD_GROUP_HIGH_SUM', 'PREV_HAS_REFUSED_SUM',
                'PREV_HAD_LATE_PAYMENTS_MEAN', 'ACTIVE_REMAINING_DEBT_SUM', 'ACTIVE_AMT_INSTALMENT_SUM',
                'REFUSED_APP_CREDIT_PERC_MEAN', 'APPROVED_AMT_DOWN_PAYMENT_SUM', 'REFUSED_DAYS_DECISION_MEAN',
                'POS_SK_DPD_MEAN', 'POS_SK_DPD_DEF_MEAN', 'POS_COUNT', 'INSTAL_DPD_MEAN', 'INSTAL_DBD_MEAN',
                'INSTAL_PAYMENT_PERC_MEAN', 'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_AMT_PAYMENT_MEAN',
                'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'INSTAL_COUNT', 'CC_MONTHS_BALANCE_MEAN', 'CC_AMT_BALANCE_MEAN',
                'CC_CNT_DRAWINGS_CURRENT_MEAN', 'CC_LIMIT_USE_MEAN', 'CC_LATE_PAYMENT_MEAN', 'CC_DRAWING_LIMIT_RATIO_MEAN']



app = FastAPI()

origins = ["*"] # specifier le domaine depuis lequel les requetes peuvent venir

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    AMT_ANNUITY : float
    DAYS_BIRTH : float
    DAYS_EMPLOYED : float
    DAYS_ID_PUBLISH : float
    REGION_RATING_CLIENT_W_CITY : float
    HOUR_APPR_PROCESS_START : float
    EXT_SOURCE_1 :  float
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    CREDIT_TO_ANNUITY_RATIO : float
    CREDIT_TO_GOODS_RATIO : float
    ANNUITY_TO_INCOME_RATIO : float
    INCOME_TO_EMPLOYED_RATIO : float
    INCOME_TO_BIRTH_RATIO : float
    NAME_EDUCATION_TYPE_CATed : float
    BURO_DAYS_CREDIT_MEAN : float
    BURO_CREDIT_ENDDATE_MAX : float
    BURO_AMT_CREDIT_MAX_OVERDUE_MEAN : float
    BURO_AMT_CREDIT_SUM_LIMIT_MEAN :  float
    BURO_DEBT_PERCENTAGE_MEAN : float
    BURO_HAS_CREDIT_SUM : float
    BURO_HAS_CLOSED_SUM : float
    BURO_HAS_DELAYED_BADDEBT_MEAN : float
    PREV_CREDIT_TO_ANNUITY_RATIO_MEAN : float
    PREV_SIMPLE_INTERESTS_MEAN : float
    PREV_DOWN_PAYMENT_TO_CREDIT_MEAN : float
    PREV_DAYS_LAST_DUE_DIFF_MEAN : float
    PREV_YIELD_GROUP_HIGH_SUM : float
    PREV_HAS_REFUSED_SUM : float
    PREV_HAD_LATE_PAYMENTS_MEAN : float
    ACTIVE_REMAINING_DEBT_SUM : float
    ACTIVE_AMT_INSTALMENT_SUM : float
    REFUSED_APP_CREDIT_PERC_MEAN : float
    APPROVED_AMT_DOWN_PAYMENT_SUM : float
    REFUSED_DAYS_DECISION_MEAN : float
    POS_SK_DPD_MEAN : float
    POS_SK_DPD_DEF_MEAN : float
    POS_COUNT : float
    INSTAL_DPD_MEAN : float
    INSTAL_DBD_MEAN : float
    INSTAL_PAYMENT_PERC_MEAN : float
    INSTAL_PAYMENT_DIFF_MEAN : float
    INSTAL_AMT_PAYMENT_MEAN : float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN : float
    INSTAL_COUNT : float
    CC_MONTHS_BALANCE_MEAN : float
    CC_AMT_BALANCE_MEAN : float
    CC_CNT_DRAWINGS_CURRENT_MEAN : float
    CC_LIMIT_USE_MEAN : float
    CC_LATE_PAYMENT_MEAN : float
    CC_DRAWING_LIMIT_RATIO_MEAN : float
    

# loading the saved model
xgb_model = pickle.load(open('model.pkl','rb'))


@app.post('/solvability_prediction')
def solvability_prediction(input_parameters : model_input):
    input_list = []

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    for feature in REF_FEATURES : 
        input_list.append(input_dictionary[feature])
    
    prediction = xgb_model.predict([input_list])
    
    if prediction[0] == 0:
        return 0  
    else:
        return 1
