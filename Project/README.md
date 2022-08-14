
# Problem Statement
*Business Need*
You work at Rossmann Pharmaceuticals as a data scientist. The finance team wants to forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgement to forecast sales.

The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.

Your job is to build and serve an end-to-end product that delivers this prediction to Analysts in the finance team.

*Goals*
Perform Exploratory Data Analysis
Preprocess Data with Sklearn
Build models with SKlearn pipelines
Serialize models and serve it to a web interface with Flask

# Model Development
The model is developed using Random forest regressor to predict the sales across multiple stores. 

mlflow is used to montor the experiment and prefect is used as the workflow orchestration tool



# Model deployment

The model is deployed in two ways:
 -1 Self serving using Kserve
 -2 Webservice using docker and flask

 *-1 Self serving inference service*
 A custom sklearn service is built by adapting the codes in the kserve repository 
  To do this, clone the repository and copy the sklearn.Dockerfile from inferenceservice folder in this repo to the kserve-master/python
  Build the docker image
  Use the docker image with the inference.yaml provided in the imferencservice folder. 
  A prebuilt docker image is provided in inference.yaml, this can be used as well. 
  Adding the location of the model
  Goto the directory with the model and 
  run python -m http.server
  This will give a webserve providing the content of the directory. Goto this link and copy the link to the model.pkl. Since kserve will not recognise the localhost address. On your terminal, enter ifconfig(linux) or ipconfig(windows) and copy your ip address. Replace the local host with this ip address and build the service using kubectl
 
 *-2 Webservice using Flask and Docker*
 cd to webservice folder
 Build docker image using
    sudo docker build -t sales-prediction-service:v1 .
Run the image with:
    sudo docker run -it --rm -p 9696:96 sales-prediction-service:v1
    
Run the test with:
change the local host in the url to your localhost address
    run:
    Python test.py