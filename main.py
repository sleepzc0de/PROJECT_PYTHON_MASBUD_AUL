import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import joblib


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neighbors import  KNeighborsRegressor, KNeighborsClassifier

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease


df_sbk = pd.read_excel('SBSK_GK_KKEFI.xlsx')
# cleansing data mas bud kolom  dan tipe bangunan namanya ga cocok
df_sbk = df_sbk.rename(columns={"Tipe Bangunan\n(isi dengan angka 1 - 3)": "tipe_bangunan"})
df_sbk.columns = df_sbk.columns.str.replace(' ', '_').str.lower()
df_pivot = df_sbk.pivot_table(
    index="kode_eselon_i",
    columns='tipe_bangunan',
    values='luas_sbsk',
    aggfunc='sum'
)

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

X = df_sbk.drop(columns=["luas_sbsk"])
y = df_sbk["luas_sbsk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('numeric', numerical_pipeline, ['menteri', 'wamen',
       'es_ia_kk', 'es_ia_nkk', 'es_ib', 'es_iia_kk', 'es_iia_nkk', 'es_iib',
       'es_iii_kk', 'es_iii_nkk', 'es_iv_kk', 'es_iv_nkk', 'es_v', 'f-iv',
       'f-iii', 'pelaksana', 'jumlah_pegawai', 'jumlah_pengunjung',
       'luas_gk_eksisting', 'rkerja', 'rarsip',
       'r_fungsional', 'toilet', 'r_server', 'r_layanan', 'lobby', 'nisbah']),
    ('categoric', categorical_pipeline, ['kode_eselon_i', 'kode_korwil', 'tipe_kantor', 'tipe_bangunan'])])

# print(preprocessor)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor())
])

pipeline.fit(X_train, y_train)

pipeline.score(X_train, y_train)

# DUMP PIPLINE
joblib.dump(pipeline, 'ml_pipeline.joblib')

