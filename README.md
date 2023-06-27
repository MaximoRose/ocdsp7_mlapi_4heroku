# ocdsp7_mlapi_4heroku

Repo dédié à l'application Heroku hostant l'API.

1. __complementary_functions.py :__ deux fonctions dédiées au preprocessing et à SHAP Force
2. __main.py :__ La déclaration des API à proprement parler
3. __mbr_kernel.py :__ Les fonctions nécessaires aux préptraitements des données métiers
4. __MLModel :__ Le fichier issu de MLFlow décrivant notre modèle
5. __model.pkl :__ Le modèle de production issu du registre MLFLow
6. __Procfile :__ Fichier propre à heroku, permettant l'exécution de l'app
7. __requirements.txt & runtime.txt :__ Fichiers déclarant l'environnement nécessaires pour l'exécution du code

