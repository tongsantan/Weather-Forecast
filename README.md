# Weather Forecast Project

## How to run?
### STEPS:

Clone the repository

```bash
https://github.com/tongsantan/Weather-Forecast.git
```
### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -n mlenv python=3.8 -y
```

```bash
conda activate mlenv
```

### STEP 02 - create file structure using template
```bash
python template.py
```

### STEP 03 - install the requirements
```bash
pip install -r requirements.txt
```

### STEP 04 - Execute data ingestion, transformation and training

```bash
# Run the following command
python main.py
```

### STEP 05 - Model Prediction 

```bash
# Finally run the following command
python app.py
```

```bash
# open up your local host and port
# enter the details for Model Prediction
```

### STEP 06 - Model Deployment 

```bash
# Building a docker image
docker build -t mldockerimg:v1 .

## Running a docker image using your desired port number XXXX:XXXX
docker run -dp XXXX:XXXX -ti --name mlContainer mldockerimg:v1
```




