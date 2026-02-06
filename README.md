# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and extract the independent variables X and dependent variable Y.

2. Split the dataset into training and testing sets.

3. Initialize the SGD classifier parameters such as learning rate, maximum iterations, and tolerance.

4. Compute the linear model output using:

<img width="124" height="59" alt="image" src="https://github.com/user-attachments/assets/33f8bc25-2823-4cf9-94da-3d1599cf24e5" />


5. Update the parameters using stochastic gradient descent:
<img width="285" height="121" alt="image" src="https://github.com/user-attachments/assets/7c6bdc54-6876-40c6-b743-3321e980c039" />

where the gradient is computed using one training sample at a time.

6. Use the trained model to predict class labels and evaluate performance using accuracy and confusion matrix.

## Program:
```PYTHON
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SRI SRINIVASAN K
RegisterNumber:  212224220104
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

```

## Output:

<img width="791" height="405" alt="image" src="https://github.com/user-attachments/assets/464de948-ec09-4652-8175-b4034e1041c4" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
