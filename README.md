# mlflow-for-mlops
This repo shows how to use [MLflow](https://mlflow.org/) to track end to end a machine learning project. It can be used to understand how a typical MLOps pipeline should look like.

Covered topics:
* Experiment tracking
* Model tracking / registry
* Model serving (online/batch)

# Tracking server & Experimentation
The first step is to start the MLflow tracking server, this can be done by running the following script:

```bash
./start_tracking_server.sh
```

After the tracking seerver is correctly running, it is possible to run the **experiment.py** script, it will perform the following steps:

* Train a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model for multiclass classification on the [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). The output of the model can be on of the **three classes** of the training dataset (class 0, class 1, class 2)
* Track the experiment (hyperparameters and metrics). Learn more [here](https://mlflow.org/docs/latest/tracking.html)
* Register the trained model in the [Model Registry](https://mlflow.org/docs/latest/model-registry.html)

# Serving

A model is usually served in two possible ways:

* Batch: usually for non-interactive and scheduled jobs
* Online: usually for streaming or rest-api based scenarios

**IMPORTANT**: Be sure the tracking server is running before running the serving scripts! Both the examples pull the model from the tracking server

The examples can be found under the **serving** directory:

* **batch**: a simple example that shows how to predict using the pulled model
* **online**: [FastAPI](https://fastapi.tiangolo.com/) based rest service that can be used to serve the model via a standard HTTP endpoint 
