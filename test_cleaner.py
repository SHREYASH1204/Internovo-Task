import os
import csv
import json
from fastapi.testclient import TestClient
from app import app, CSV_FILE, CSV_HEADERS, init_csv, save_to_csv

def run_tests():
    print("=== Starting propOG Listing Cleaner Verification ===")
    
    # 1. Test CSV initialization and strict column structure
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    init_csv()
    
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    
    assert header == ["bhk", "property_type", "locality", "area_sqft"], f"Unexpected headers: {header}"
    print("[PASS] CSV headers strictly match ['bhk', 'property_type', 'locality', 'area_sqft'] and no other columns.")
    
    # 2. Test saving structured records to CSV
    test_record_1 = {"bhk": 2.0, "property_type": "flat", "locality": "Koramangala", "area_sqft": 1200.0}
    test_record_2 = {"bhk": 3.0, "property_type": "villa", "locality": None, "area_sqft": None}
    test_record_3 = {"bhk": None, "property_type": None, "locality": None, "area_sqft": None}
    
    save_to_csv(test_record_1)
    save_to_csv(test_record_2)
    save_to_csv(test_record_3)
    
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    
    assert len(rows) == 4, f"Expected 4 rows (1 header + 3 data), got {len(rows)}"
    assert rows[1] == ["2.0", "flat", "Koramangala", "1200.0"]
    assert rows[2] == ["3.0", "villa", "", ""]
    assert rows[3] == ["", "", "", ""]
    print("[PASS] save_to_csv correctly wrote numeric and null values without extra columns.")
    
    # 3. Test API endpoint with FastAPI TestClient
    client = TestClient(app)
    
    # Reset CSV for clean test
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    init_csv()
    
    print("\nTesting API with full note (0 missing fields)...")
    res1 = client.post("/api/clean-listing", json={
        "raw_description": "2bhk flat for sale in koramangala 1200 sqft urgent sale prime loc near metro"
    })
    assert res1.status_code == 200, f"res1 failed: {res1.text}"
    data1 = res1.json()
    assert data1["bhk"] == 2.0 or data1["bhk"] == 2
    assert data1["property_type"] == "flat"
    assert "koramangala" in data1["locality"].lower()
    assert data1["area_sqft"] == 1200.0 or data1["area_sqft"] == 1200
    assert data1["needs_more_info"] is False
    print("[PASS] Full note: extracted all 4 fields correctly, needs_more_info is False.")
    
    print("\nTesting API with partial note triggering >2 missing fields (needs_more_info)...")
    res2 = client.post("/api/clean-listing", json={
        "raw_description": "urgent sale flat available immediately no brokerage prime building"
    })
    assert res2.status_code == 200, f"res2 failed: {res2.text}"
    data2 = res2.json()
    assert data2["property_type"] == "flat"
    assert data2["bhk"] is None
    assert data2["locality"] is None
    assert data2["area_sqft"] is None
    assert len(data2["missing_fields"]) == 3
    assert data2["needs_more_info"] is True
    print(f"[PASS] Note with 3 missing fields flagged as needs_more_info=True: missing_fields={data2['missing_fields']}")

    # 4. Verify CSV content after API calls
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
        api_csv_rows = list(csv.reader(f))
    
    assert len(api_csv_rows) == 3 # 1 header + 2 api calls
    assert api_csv_rows[0] == ["bhk", "property_type", "locality", "area_sqft"]
    print("[PASS] CSV on disk verified with only the 4 required fields.")
    
    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
