import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("mental_health_prediction.csv")

final_features = [
    "sleep_hours", "social_media_hours", "academic_work_pressure",
    "physical_activity_days", "stress_level", "anxiety_score",
    "depression_score", "work_life_balance", "mood_score",
    "concentration_level", "social_support"
]

X = df[final_features]
y = df["mental_health_condition"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(C=0.1, max_iter=1000, random_state=42),
)
model.fit(X_train, y_train)

joblib.dump(model, "mental_health_model.joblib")
print("saved: mental_health_model.joblib")
