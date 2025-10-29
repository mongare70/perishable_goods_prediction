import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .section-description {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #20b2aa 0%, #32cd32 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
        border-radius: 0.5rem !important;
        border: none !important;
        margin-top: 2rem !important;
        margin-bottom: 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1a9a94 0%, #28a728 100%) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        transform: translateY(-1px) !important;
    }
    .stButton>button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-title">📦 Perishable Goods Prediction</h1>
        <p class="subtitle">Forecast your product sales with precision using advanced analytics</p>
    </div>
    """, unsafe_allow_html=True)

with st.form("form"):
    # Product Details Section
    st.markdown("""
        <div>
            <h3 class="section-header">✨ Product Details</h3>
            <p class="section-description">Enter your product and store information to get an accurate sales prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Product Information Subsection
    st.markdown("""
        <div>
            <h4 class="subsection-header">📋 Product Information</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Two-column layout for Product Information
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input("Product Name", value="Whole Meal Bread 600g")
        product_category = st.selectbox(
            "Product Category",
            ["Bakery", "Beverages", "Dairy", "Meat"],
        )
    
    with col2:
        wastage_units = st.number_input("Wastage Units", value=100, min_value=0)
        price = st.number_input("Price (£)", value=3.3, min_value=0.0, format="%.2f")
    
    # Shelf Life slider (full width)
    shelf_life_days = st.slider("Shelf Life (Days)", min_value=1, max_value=365, value=1, step=1)
    
    # Store & Location Information Section
    st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 class="section-header">📈 Store & Location Information</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Two-column layout for Store & Location Information
    col3, col4 = st.columns(2)
    
    with col3:
        cold_storage_capacity = st.number_input(
            "Cold Storage Capacity", value=500, min_value=0
        )
        store_size = st.number_input("Store Size (sq ft)", value=1600, min_value=0)
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
    
    with col4:
        week_number = st.text_input("Week Number", value="2024-W33")
        rainfall = st.number_input("Rainfall (mm)", value=23.3, min_value=0.0, format="%.1f")
        avg_temp = st.number_input("Average Temperature (°C)", value=21.1, min_value=0.0, format="%.1f")
    
    # Generate Prediction Button - positioned at the bottom
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("📈 Generate Prediction", use_container_width=True)


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

        api_url = f"{os.getenv('host', '127.0.0.1')}/predict"
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
