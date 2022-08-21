
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

# Run the project
Install the requirements in the project directory with 

``` shell
pip install -r requirement.txt
```
Run the training scripts with:
``` shell
Python train.py
```
This will train the model and monitor the experiemnt. Prefect is configured to do this every 5 minutes. 
Register the model in mlflow model registry
``` shell
Python register_model.py
```
This will register the model with the lowest error

# Model Development
The model is developed using Random forest regressor to predict the sales across multiple stores. 

mlflow is used to montor the experiment and prefect is used as the workflow orchestration tool.

Evidently is configured to monitor the perfromance of the model 



# Model deployment

The model is deployed in two ways:
 -1 Self serving using Kserve
 -2 Webservice using docker and flask

 ### Self serving inference service
<p>A custom sklearn service is built by adapting the codes in the kserve repository (https://github.com/kserve/kserve). To do this, clone the repository and copy the sklearn.Dockerfile from inferenceservice folder in this repo to the kserve-master/python.</p>
<p>Build the docker image. Use the docker image with the inference.yaml provided in the imferencservice folder. A prebuilt docker image is provided in inference.yaml, this can be used as well. 
 #### Adding the location of the model to the inference yaml
 <p>Goto the directory with the model and </p>
 ``` shell
  run python -m http.server
  ```
 <p>This will give a webserver providing the content of the directory. Goto this link and copy the link to the model.pkl. Since kserve will not recognise the localhost address. On your terminal, enter ifconfig(linux) or ipconfig(windows) and copy your ip address. Replace the local host with this ip address and build the service using kubectl </p>
 <p>Test script using the test.py script in /script. Remember to chaange the host address</p>
 
 ### Webservice using Flask and Docker
 
 Build docker image using
 ``` shell
    sudo docker build -t sales-prediction-service:v1 .
 ```
Run the image with:
``` shell
    sudo docker run -it --rm -p 9696:9696 sales-prediction-service:v1
```
    
Run the test with:
change the local host in the url to your localhost address
    run:
``` shell
    Python test.py
```
### Testing frame work
<p> Pytest is used to developed an integration test for the dockerised webservice. The details can be found in the integration test folder. This can be run with run.sh file</p>

### Alternatively
The project can be run using the makefile in this project directory

Thanks!
