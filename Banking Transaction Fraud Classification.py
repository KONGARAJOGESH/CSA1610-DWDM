import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("bank_transactions.csv")

df = df.drop_duplicates()

num = df.select_dtypes(include=['int64','float64']).columns
cat = df.select_dtypes(include=['object']).columns

df[num] = SimpleImputer(strategy='median').fit_transform(df[num])

for c in cat:
    df[c] = SimpleImputer(strategy='most_frequent').fit_transform(df[[c]]).ravel()

le = LabelEncoder()

for c in cat:
    df[c] = le.fit_transform(df[c].astype(str))

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf")
}

for name, model in models.items():

    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    print("\n", name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1 Score :", round(f1_score(y_test, pred, zero_division=0), 4))
