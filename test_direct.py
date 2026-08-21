import sys
from app import _call_single_ai_attempt, call_ai_api_with_retry, CSV_FILE, init_csv, save_to_csv
import csv
import os

print("Testing direct AI call...", flush=True)
try:
    res = _call_single_ai_attempt("2bhk flat for sale in koramangala 1200 sqft urgent sale prime loc near metro")
    print(f"Result: {res.model_dump()}", flush=True)
except Exception as e:
    print(f"Error in direct AI call: {e}", flush=True)
