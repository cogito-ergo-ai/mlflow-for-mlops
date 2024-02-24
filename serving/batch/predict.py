import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:8080")
logged_model = "models:/iris_model/1"  #

loaded_model = mlflow.pyfunc.load_model(logged_model)
test_sample = [2.3, 1.0, 3.4, 6.7]
predicted = loaded_model.predict(pd.DataFrame([test_sample]))

print(f"test_sample: {test_sample} | predicted: {predicted}")
