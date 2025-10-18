import pandas as pd
import numpy as np


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """This function is used to clean incoming data"""
    product_categories = ["Bakery", "Beverages", "Dairy", "Meat"]
    regions = [
        "London",
        "Midlands",
        "North East",
        "North West",
        "South East",
        "South West",
    ]

    nominal_columns = ["Product_Category", "Region"]
    high_cardinality_columns = ["Product_Name"]

    for col in nominal_columns:
        if col in data.columns:
            dummies = pd.get_dummies(data[col], prefix=col)
            dummies.astype(int)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)

    for col in high_cardinality_columns:
        mean_encode = data.groupby(col)["Price"].mean()
        data[col] = data[col].map(mean_encode)

    # convert WeekNumber into datetime format then to something appropriate for mutual_information_regression eg(20240101)
    data["Week_Number"] = pd.to_datetime(
        data["Week_Number"].astype(str) + "-1", format="%G-W%V-%u"
    )

    # Convert to yyyymmdd integer format, e.g. 20240101
    data["Week_Number"] = data["Week_Number"].dt.strftime("%Y%m%d").astype(int)

    expected_dummies = [f"Product_Category_{cat}" for cat in product_categories] + [
        f"Region_{r}" for r in regions
    ]

    for col in expected_dummies:
        if col not in data.columns:
            data[col] = 0

    expected_columns = [
        "Wastage_Units",
        "Product_Name",
        "Shelf_Life_Days",
        "Price",
        "Cold_Storage_Capacity",
        "Store_Size",
        "Rainfall",
        "Avg_Temperature",
        *expected_dummies,
        "Week_Number",
    ]

    # keep only relevant columns and fill missing ones with 0
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_columns]

    return data
