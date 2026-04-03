# Meet our ML TEAM!!

- Vrajesh Sharma - ML Head
- Sharv Mehta - ML Lead
- Samarth Srivastava - ML Lead
- Saurabh Singh - ML Lead
- Mohit Manglani - ML Lead
- Dhruval Patel - ML Lead


## Follow Me On Kaggle for Resources
Every notebook and dataset will be provided on this profile - [Vrajesh Sharma](https://www.kaggle.com/vrajesh0sharma7)


### Team Formation
Link for team registration - [Link](https://forms.gle/mn6wPEz1bk6bUjon8)


---
# Session 2
- Kaggle Notebook - [Salary Prediction](https://www.kaggle.com/code/vrajesh0sharma7/aspdc-ann-salary-prediction)
- Dataset - [Salary Dataset](https://www.kaggle.com/datasets/vrajesh0sharma7/salary-prediction-250k)
- Handwritten Notes is available in **Session 2** folder.

---
# Session 3
- Kaggle Notebook - [Person Classification](https://www.kaggle.com/code/vrajesh0sharma7/aspdc-cnn-person-classification)
- Dataset - [Facial Dataset](https://www.kaggle.com/datasets/vrajesh0sharma7/aspdc-cnn-images)
- Presentation Slides will be uploaded soon.

---
# Session 4
- Kaggle Notebook - [Stock Price Prediction](https://www.kaggle.com/code/vrajesh0sharma7/aspdc-rnn-stock-price-prediction)
- Dataset - [NIFTY50](https://www.kaggle.com/datasets/vrajesh0sharma7/nifty50)

---

# Session 5

### main.py
This is the main server file that handles all the endpoints.
```python
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Load Model
model = load_model("app/salary_ann_model.keras")

# Load scaler
with open("app/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load encoders
with open("app/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load feature column order
with open("app/columns.pkl", "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)

# FastAPI App
app = FastAPI(
    title="Salary Prediction API",
    version="1.0"
)

# Input Schema
class InputData(BaseModel):
    data: dict


# Routes
@app.get("/")
def home():
    return {
        "message": "Salary ANN API Running 🚀",
        "endpoints": ["/predict", "/author"]
    }

@app.get("/author")
def author():
    return {
        "name": "Vrajesh Sharma",
        "role": "AI-ML Engineer | ASPDC ML Head"
    }

@app.post("/predict")
def predict(input_data: InputData):
    try:
        data = input_data.data

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Validate all required columns
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}

        # Apply Label Encoding
        for col, le in label_encoders.items():
            try:
                df[col] = le.transform(df[col])
            except:
                return {"error": f"Invalid value for column: {col}"}

        # Ensure correct column order
        df = df[FEATURE_COLUMNS]

        # Apply Scaling
        X_scaled = scaler.transform(df)

        # Prediction
        pred = model.predict(X_scaled)

        # Extract value safely
        result = float(pred[0][0])

        return {
            "predicted_salary": result,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }
```

### requirements.txt
This file has all the dependancies or the libraries.
```txt
fastapi
uvicorn
tensorflow
numpy
pandas
scikit-learn
pydantic
```

### Dockerfile
This is the heart and soul of any deployment
```docker
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Folder Structure
```txt
project/ 
│ 
├── app/ 
│ ├── main.py # FastAPI backend 
│ ├── salary_ann_model.keras # Trained ANN model 
│ ├── scaler.pkl # StandardScaler 
│ ├── label_encoders.pkl # LabelEncoders 
│ ├── columns.pkl # Feature column order 
├── requirements.txt # Python dependencies 
├── Dockerfile # Deployment configuration 
├── README.md # Project documentation
```
### For any query or support 
Contact Vrajesh - 9429619257