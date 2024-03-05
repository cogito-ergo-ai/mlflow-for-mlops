import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

EXPERIMENT_NAME = "iris_model"
RUN_NAME = "iris_model_run"
ARTIFACT_PATH = "iris_artifact"
client = mlflow.MlflowClient(tracking_uri="http://127.0.0.1:8080")
mlflow.set_tracking_uri("http://127.0.0.1:8080")


def experiment_exists_or_create(name):
    experiment = client.get_experiment_by_name(name)
    if experiment:
        return experiment
    return client.create_experiment(name=name, tags={})  # set tags here as you prefer


if __name__ == "__main__":
    # load the test iris dataset
    experiment = experiment_exists_or_create(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    test_size = 0.2

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    C = 0.1  # See: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn-linear-model-logisticregression
    logreg = LogisticRegression(C=C)
    logreg.fit(X, Y)
    y_pred = logreg.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    features_info = {
        "features": {
            "sep_length": "sepal length in cm",
            "sep_width": "sepal width in cm",
            "pet_length": "petal length in cm",
            "pet_width": "petal width in cm",
        }
    }

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params({"C": C, "test_size": test_size})
        mlflow.log_metrics({"accuracy": accuracy})
        mlflow.log_dict(features_info, "features_info.json")
        mlflow.sklearn.log_model(
            sk_model=logreg,
            input_example=X_val,
            artifact_path=ARTIFACT_PATH,
            registered_model_name="iris_model",
        )
