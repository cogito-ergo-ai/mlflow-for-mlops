import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:8080")
logged_model = "runs:/68a265374b8746608394ad51ed8bb0ef/iris_artifact" # replace with an existing experiment ID

loaded_model = mlflow.pyfunc.load_model(logged_model)
test_sample = [2.3, 1.0, 3.4, 6.7]
predicted = loaded_model.predict(pd.DataFrame([test_sample]))

print(f"test_sample: {test_sample} | predicted: {predicted}")
