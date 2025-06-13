import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]


def preprocess_data(input_path, output_path):
    # meload data dan mengopy untuk memastikan datanya tidak rusak
    df = pd.read_csv(input_path)
    df_processed = df.copy()

    # menghapus missing value
    df_cleaned = df_processed.dropna()
    
    # menghapus data duplikat
    df_cleaned = df_cleaned.drop_duplicates()

    # Menangani nilai 0 pada restingBP dan cholesterol dengan mengganti dengan NAN
    df_cleaned["RestingBP"].replace(0, pd.NA, inplace=True)
    df_cleaned["Cholesterol"].replace(0, pd.NA, inplace=True)

    # Mengisi nilai NAN dengan nilai median
    df_cleaned["RestingBP"].fillna(df_cleaned["RestingBP"].median(), inplace=True)
    df_cleaned["Cholesterol"].fillna(
        df_cleaned["Cholesterol"].median(), inplace=True
    )

    # menghapus nilai outlier dengan memanggil fungsi remove_outliers_iqr
    for col in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
        before = df_cleaned.shape[0]
        df_cleaned = remove_outliers_iqr(df_cleaned, col)
        after = df_cleaned.shape[0]
        print(f"{col}: removed {before - after} outliers")

    # mengubah nilai categorikal menjadi numerik
    categorical_cols = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])

    # mengubah nilai numerik menjadi nilai yang dapat digunakan untuk proses prediksi
    numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    scaler = RobustScaler()
    df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])
    
    # menyimpan hasil prepricessing ke nama yang dituju
    df_cleaned.to_csv(output_path, index=False)
