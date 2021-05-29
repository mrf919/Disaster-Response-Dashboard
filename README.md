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
## Project Motivation
being able to reliably distinguish the disasters in the emergency and its category is vital to able to properly help the people on time. To be able to do this, is simple but efficient dashboard designed, which is able to be fed with the massage and it gives the categories.

## File Descriptions
This project consists of three main parts:
-	ETL Pipeline:

    This pipeline performs the Extract, Transform, and Load process to prepare the learning data as clean input for the Machine learning process. This data will be saved in a      SQLite database. This pipeline can be found in [process_data.py](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/process_data.py).

-	ML Pipeline:

    Using the NLTK method and multi-output classification, this pipeline uses the massages to predict their categories. This pipeline can be found in [train_classifier.py](https://github.com/mrf919/Disaster-Response-Dashboard/blob/main/train_classifier.py).

-	Flask Web app:

    The flask web app provides the web-based user interface which is connected with the database and pipelines and generate the visualisations.
 


## Results
The findings and the resualts of the code is availble [here](https://medium.com/@m.r.farhood/is-there-any-trend-change-in-tools-used-c4cbb41d4710)

## Licensing, Authors, Acknowledgements
The findings and the results of the code is available [here](https://www.kaggle.com/haakakak/stack-overflow-developer-surveys-20152020)
