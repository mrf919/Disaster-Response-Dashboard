# Disaster-Response-Dashboard
This dashboard has a Machin learning pipeline in background, which analyses a data set of massages and their categories to train and based on this training it can predict the categories for a new massage and display it in the dashboard
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation
Stack overflow trend requires:
- Python (>= 3.6)
- the following libraries are needed to run the code:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - nltk
    - re
    - sklearn
    - plotly
    - json
    - flask
## Project Motivation
being able to reliably distinguish the disasters in the emergency and its category is vital to able to properly help the people on time. To be able to do this, is simple but efficient dashboard designed, which is able to be fed with the massage and it gives the categories.

## File Descriptions
This project consists of three main parts:
-	ETL Pipeline:

    This pipeline performs the Extract, Transform, and Load process to prepare the learning data as clean input for the Machine learning process. This data will be saved in a       [SQLite database](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/data/DisasterResponse.db). This pipeline can be found in [process_data.py](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/data/process_data.py).
    To perform this task, there are two .csv data needed. The [disaster_categories.csv](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/data/disaster_categories.csv) and [disaster_messages](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/data/disaster_messages.csv) which can be replaced with the .csv data having the same format and including similar infomation.

-	ML Pipeline:

    Using the NLTK method and multi-output classification, this pipeline uses the massages to predict their categories. This pipeline can be found in [train_classifier.py](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/models/train_classifier.py).

-	Flask Web app:

    The flask web app provides the web-based user interface which is connected with the database and pipelines and generate the visualisations. The [master.html](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/app/templates/master.html) includes the HTML format of the webpage and [go.html](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/app/templates/go.html) highlights the gategories regarding the searched massege.
 
## instruction
to run this project, the follosing commands should be executed in the project root directory:

1-  Runing the ETL pipeline:
  
    python data/process_data.py [messages_filepath] [categories_filepath] [database_filepath]
    
   [messages_filepath] is the path to the .csv data which contains the "disaster_messages"
   
   [categories_filepath] is the path to the .csv data which contains the "disaster_categories"
   
   [database_filepath] is the path to the .db data which contains the which is going to stor the data
   
2- Runing the ML pipeline:

    python models/train_classifier.py [database_filepath] models/classifier.pkl
    
    
   [database_filepath] is again the same path to the .db data which contains the which is going to store the data
   
3- Starting the web app

    python app/run.py [database_filepath]
    
   [database_filepath] is again the same path to the .db data which contains the which is going to store the data
    
4- The connection link:

    env|grep WORK
    
   (in a new Terminal)
   
the connection link consists of 

    https://SPACEID-3001.SPACEDOMAIN


## Licensing, Authors, Acknowledgements
The findings and the results of the code is available [here](https://www.kaggle.com/haakakak/stack-overflow-developer-surveys-20152020)
