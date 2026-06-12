import requests
import time
import pandas as pd
import random

URL = "http://127.0.0.1:8000/predict"

test_inputs = [
    {"recency": 5, "frequency": 30, "avg_order_value": 400},
    {"recency": 10, "frequency": 20, "avg_order_value": 120},
    {"recency": 7, "frequency": 25, "avg_order_value": 20},
    {"recency": 40, "frequency": 3, "avg_order_value": 500},
    {"recency": 2, "frequency": 1, "avg_order_value": 60},
    {"recency": 180, "frequency": 2, "avg_order_value": 80},
    {"recency": 1, "frequency": 1, "avg_order_value": 80},
    {"recency": 364, "frequency": 1, "avg_order_value": 40},
    {"recency": 60, "frequency": 8, "avg_order_value": 100},
    {"recency": 120, "frequency": 20, "avg_order_value": 150},
    {"recency": 1, "frequency": 100, "avg_order_value": 200},
    {"recency": 15, "frequency": 15, "avg_order_value": 1000},
    {"recency": 50, "frequency": 2, "avg_order_value": 1000},
    {"recency": 70, "frequency": 1, "avg_order_value": 1500}
    
]

print("\nRunning experiments...\n")

results = []

test_number = 1

for _ in range(10):

    for features in test_inputs:
        
        features_copy = features.copy()
        
        features_copy["recency"] += random.randint(-3, 3)
        features_copy["frequency"] += random.randint(-1,1)
        features_copy["avg_order_value"] += random.randint(-20,20)

        features_copy["recency"] = max(1, features_copy["recency"])
        features_copy["frequency"] = max(1, features_copy["frequency"])
        features_copy["avg_order_value"] = max(1, features_copy["avg_order_value"])
        try:
            response = requests.post(URL, json=features_copy)

            if response.status_code == 200:
                result = response.json()

                row = {
                    **features_copy,
                    **result
                }

                results.append(row)

                print(f"Test {test_number}: SUCCESS")
                print("Input:", features_copy)
                print("Output:", result)
                print("-" * 40)
            else:
                print(
                    f"Test {test_number}: FAILED" 
                    f"({response.status_code})"
                )
    
                print(response.text)

        except Exception as e:
            print(f"Test {test_number}: ERROR")
            print(e)

        test_number += 1

        time.sleep(0.3)

df = pd.DataFrame(results)

print("\nExperiment Results:")
print(df)

print("\nCluster Distribution:")
print(df["cluster"].value_counts())

print("\nDone.")

