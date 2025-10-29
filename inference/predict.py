from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from clean.preprocess import clean_data
import pickle
import pandas as pd
from typing import Dict, List, Any
import os
import uvicorn
from dotenv import load_dotenv
import requests
from io import BytesIO

load_dotenv()
app = FastAPI(title="Perishable Goods Prediction API", version="1.0")


class Item(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        example=[
            {
                "Wastage_Units": 100,
                "Product_Name": "Whole Wheat Bread 800g",
                "Product_Category": "Bakery",
                "Shelf_Life_Days": 4,
                "Price": 3.3,
                "Cold_Storage_Capacity": 500,
                "Region": "North",
                "Store_Size": 1600,
                "Rainfall": 23.3,
                "Avg_Temperature": 21.1,
                "Week_Number": "2024-W36"
            }
        ],
    )


@app.post("/predict")
def predict(req: Item):
    try:
        data = pd.DataFrame(req.records)
        cleaned_data = clean_data(data)
        # model_path = os.path.join(os.path.dirname(__file__), "..","model", "rf_model.pkl")
        # with open(model_path, "rb") as f:
        #     model = pickle.load(f)

        response = requests.get(os.getenv("model_url_hugging_face"))
        response.raise_for_status()

        model = pickle.load(BytesIO(response.content)) 
                

        pred = model.predict(cleaned_data)

        return {"predictions": pred.tolist()}

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data Cleaning error: {str(e)}")


if __name__ == "__main__":
    print(os.getenv("port"))
    print(f"Server us running on port {os.getenv('port', 3000)}")
    uvicorn.run(app, host=os.getenv("host", "127.0.0.1"), port=int(os.getenv("port", 3000)))
