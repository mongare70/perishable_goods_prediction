import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()
st.title("Perishable goods prediction")

with st.form("form"):
    wastage_units = st.number_input("Wastage Units", value=100, min_value=0)
    product_name = st.text_input("Product Name", value="Whole Meal Bread 600g")
    product_category = st.selectbox(
        "Product Category",
        ["Bakery", "Beverages", "Dairy", "Meat"],
    )
    shelf_life_days = st.slider("Shelf Life Days", min_value=1, value=1)
    price = st.number_input("Price", value=3.3, min_value=0.0)
    cold_storage_capacity = st.number_input(
        "Cold Storage Capacity", value=500, min_value=0
    )
    store_size = st.number_input("Store Size", value=1600, min_value=0)
    rainfall = st.number_input("Rainfall", value=23.3, min_value=0.0)
    avg_temp = st.number_input("Avg Temperature", value=21.1, min_value=0.0)
    region = st.selectbox(
        "Region",
        [
            "London",
            "Midlands",
            "North East",
            "North West",
            "South East",
            "South West",
        ],
    )
    week_number = st.text_input("Week Number", value="2024-W33")

    submitted = st.form_submit_button("Predict")


if submitted:

    try:
        data = {
            "Wastage_Units": wastage_units,
            "Product_Name": product_name,
            "Product_Category": product_category,
            "Shelf_Life_Days": shelf_life_days,
            "Price": price,
            "Cold_Storage_Capacity": cold_storage_capacity,
            "Store_Size": store_size,
            "Rainfall": rainfall,
            "Avg_Temperature": avg_temp,
            "Region": region,
            "Week_Number": week_number,
        }

        api_url = f"{os.getenv("host", "127.0.0.1")}/predict"
        response = requests.post(api_url, json={"records": [data]})

        if response.status_code == 200:
            result = response.json()
            result = result.get("predictions")
            st.write(f"Estimated Units sold: {int(result[0])}")
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.write(f"Error: {e}")

if __name__ == "__main__":
    pass
