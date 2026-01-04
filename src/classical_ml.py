# src/classical_ml.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_classical_models(features, labels):
    """
    Train and evaluate classical ML models.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    results['Logistic Regression'] = classification_report(
        y_test, lr.predict(X_test_scaled), output_dict=True
    )

    # SVM
    svm = SVC(kernel='rbf')
    svm.fit(X_train_scaled, y_train)
    results['SVM'] = classification_report(
        y_test, svm.predict(X_test_scaled), output_dict=True
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results['Random Forest'] = classification_report(
        y_test, rf.predict(X_test), output_dict=True
    )

    return results
