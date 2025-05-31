import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as pyplot
import seaborn as sns

wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Gradient Boosting Classifier

# Define the parameters for the Gradient Boosting Classifier


learning_rate = 0.1
max_depth = 3
min_samples_split = 3
min_samples_leaf = 2

# setting the tracking URI 
mlflow.set_tracking_uri("https://127.0.0.1.:5000")

mlflow.set_experiment("GradientBoostingWineClassifier")

with mlflow.start_run():
    # Create and train the model
    model = GradientBoostingClassifier(learning_rate=learning_rate,
                                       max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log parameters
    
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)

    # Print classification report and confusion matrix
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    pyplot.xlabel('Predicted')
    pyplot.ylabel('True')
    pyplot.title('Confusion Matrix')
    pyplot.show()
    
    pyplot.savefig("confusion_matrix.png")
    
    
    
    
    
    