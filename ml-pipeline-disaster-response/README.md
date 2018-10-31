# Disaster Response Pipeline Project
Project to test ETL and ML pipelines and web-app deployment.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.joblib N`
    - To run ML pipeline from saved model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.joblib Y`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
* data/process_data.py: Data wrangling/ETL script
* models/train_classifier.py: Machine learning pipeline. Note that the saved file is too large to upload to git.
* run.py: Create visualization and start Flask Webapp  