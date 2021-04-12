# Simple flask API - classifier


## Onboarding

Create conda environment and activate it 

```
cd flask-simple
conda env create -f cvenv_dev.yml 
conda activate flask-simple-env
```


## Endpoints

The API has 4 end point. 

* localhost:5000/fit
    * Allowed method: POST
    * Data posted: csv file
    * Response: string (tid).
    * desc: Launch subprocess that fit passed dataset 

* localhost:5000/predict
    * Allowed method: POST
    * Data posted: csv file
    * Response: string (tid).
    * desc: Launch subprocess that predict passed dataset

* localhost:5000/track
    * Allowed method: GET
    * Args: tid
    * Response: json file
    * desc: Return cross validation result as they are computed
    
* localhost:5000/download
    * Allowed method: GET
    * Args: tid
    * Response: csv text
    * desc: return the prediction associated with tid


## Test cases

Start the flask api: Open a terminal, Get in test directory and run flask 

```
cd simple-flask
flask run
```


`test_case.sh` contains code that enable to perform action [fit, predict, track, download]. To run the test case, 
open a new terminal at the root of the project and run actions:


 * *fit* 
   ```
    # arg 2 is the name of csv file to fit model.
    # return the identifier of fitting subprocess
   
    ./test_case.sh fit data/iris.csv  
    Fitting initiated, you can track process using track command and pid: 12345
   
    ```
* predict 
  ```
    # arg 2 is the name of csv file to fit model.
    # return the identifier of fitted model used
   
    ./test_case.sh predict data/iris.csv 12345 
    Prediction initiated, you can download preds using download command and tid: 12345 
   
  ```
* track 
   ```
    # arg 2 is the name of csv file to fit model.
    # return the name of file where info can be found
   
    ./test_case.sh track 12345  
    Info fitting downloaded in 12345.json
   
  ``` 

* download 
   ```
    # arg 2 is the name of csv file to fit model.
    # return the name of file where prediction can be found
   
    ./test_case.sh download 12345  
    Prediction downloaded at 12345.csv 
   
  ``` 
