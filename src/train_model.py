import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# =====================
# LOAD DATA
# =====================

df = pd.read_csv("data/toyota.csv")


# =====================
# FITUR DAN TARGET
# =====================

X = df.drop("price", axis=1)
y = df["price"]


# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================
# KOLOM
# =====================

categorical_features = [
    "model",
    "transmission",
    "fuelType"
]

numerical_features = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]


# =====================
# PREPROCESSOR
# =====================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)


# =====================
# PIPELINE
# =====================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])


# =====================
# GRID SEARCH
# =====================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("Training model...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


# =====================
# EVALUASI
# =====================

predictions = best_model.predict(X_test)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\n===== HASIL =====")
print("Best Parameters:")
print(grid_search.best_params_)

print("\nRMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# =====================
# SIMPAN MODEL
# =====================

joblib.dump(
    best_model,
    "models/toyota_price_model.pkl"
)

print("\nModel berhasil disimpan!")