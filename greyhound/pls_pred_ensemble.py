import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.cross_decomposition import PLSRegression
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('../data/data_final.csv')

# Modify target variable for position prediction
data['Position'] = data['Finished'].apply(lambda x: min(x, 6))

# Preprocess inputs function
def preprocess_inputs(df, target='Finished', n_components=10, degree=2):
    df = df.copy()
    df = df.drop(['Race_ID', 'Finished', 'Winner'], axis=1, errors='ignore')

    if target == 'Winner':
        y = df['Winner']
        X = df.drop('Winner', axis=1)
    elif target == 'Finished':
        y = df['Finished']
        X = df.drop('Finished', axis=1)
    elif target == 'Position':
        y = df['Position']
        X = df.drop(['Position'], axis=1)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, train_size=0.7, shuffle=True, random_state=1)

    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Apply PLS regression
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, pls

# Preprocess data for position prediction
X_train, X_test, y_train, y_test, pls = preprocess_inputs(data, target='Position', n_components=10)

# Define models
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Neural Network": MLPClassifier(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

# Train and evaluate individual models
for name, model in models.items():
    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test) * 100
    print(f"{name} Test Accuracy: {test_accuracy:.2f}%")

# Voting Classifier
voting_clf = VotingClassifier(estimators=list(models.items()), voting='soft')
voting_clf.fit(X_train, y_train)
voting_accuracy = voting_clf.score(X_test, y_test) * 100
print(f"Voting Classifier Test Accuracy: {voting_accuracy:.2f}%")

# Stacking Classifier
stacking_clf = StackingClassifier(estimators=list(models.items()), final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)
stacking_accuracy = stacking_clf.score(X_test, y_test) * 100
print(f"Stacking Classifier Test Accuracy: {stacking_accuracy:.2f}%")

# Market prediction accuracy
market_data = list(zip(data['Public_Estimate'], data['Finished']))
total = len(market_data)
correct_predictions = sum([1 for estimate, actual in market_data if estimate == actual])
market_accuracy = correct_predictions / total * 100
print(f'Market Accuracy: {market_accuracy:.2f}%')
