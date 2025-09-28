import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

llcp2023 = pd.read_csv("2023SağlıkVerileri.csv")

#######################################################################
# Veri Ön İşleme
#######################################################################

df23 = llcp2023.copy()
df23.shape

# Görüşmeyi tamamlamamış olan örnekler ve gereksiz değişkenler veri setinden düşürüldü
df23 = df23[(df23['DISPCODE'] != 1200)]
df23 = df23.drop(columns=["SEQNO", "FMONTH", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "CTELENM1",
                          "CTELNUM1", "STATERE1", "CSTATE1", "CELPHON1", "CELLFON5", "IDATE",
                          "PVTRESD1", "COLGHOUS", "LADULT1", "NUMADULT", "LANDSEX2", "PVTRESD3",
                          "CCLGHOUS", "CADULT1", "HHADULT", "CELLSEX2"])
df23.shape

# Önemli türetilmiş değişkenler kopyalandı
df23["AGECAT"] = df23["_AGEG5YR"]
df23["RACE"] = df23["_RACEGR3"]

# _ içeren, türetilmiş değişkenler veri setinden çıkartıldı
columns_to_drop = [col for col in df23.columns if '_' in col] # (356261, 245)
columns_to_drop
df23.drop(columns=columns_to_drop, inplace=True)
df23.shape

# Missing değerlerin analizi, missing değerleri yakalamak için fonksiyon
def report_missing(dataframe):
  missing_counts = dataframe.isnull().sum()
  missing_percentages = 100 * dataframe.isnull().sum() / len(dataframe)
  missing_report = missing_percentages[missing_percentages > 0.0000].to_frame(name='Missing Percentage')
  missing_report["Missing Count"] = missing_counts
  missing_columns = missing_report.index.tolist()
  return missing_counts, missing_report, missing_percentages, missing_columns

# Belli bir yüzdenin üzerinde missing değerleri bulunan değişkenleri düşürmek için fonksiyon
def drop_missing(df, threshold=60, exclude=None):
    if exclude is None:
        exclude = []
    missing_percentages = df.isnull().mean() * 100
    cols_to_drop = missing_percentages[(missing_percentages > threshold) & (~missing_percentages.index.isin(exclude))].index
    df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

missing_report.shape

# %60 üzerinde missing değeri bulunan değişkenler veri setinden çıkartıldı
df23, cols_to_drop = drop_missing(df23, threshold=60)
df23.shape

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

# 18 yaşının altıdaki gözlemler veri setinden çıkartılmıştır. Gereksiz değişkenler temizlenmiştir
df23 = df23[(df23['AGECAT'] != 14)]
df23 = df23.drop(columns=['SAFETIME' , 'LANDLINE' , 'POORHLTH' , 'PRIMINS1', 'PERSDOC3' ,
                          'MEDCOST1' , 'CHECKUP1' , 'STRENGTH' , 'MARITAL' , 'EDUCA' ,
                          'RENTHOM1' , 'CPDEMO1C' , 'VETERAN3' , 'EMPLOY1' , 'CHILDREN' ,
                          'INCOME3' , 'DECIDE' , 'DIFFDRES' , 'DIFFALON' , 'FALL12MN' ,
                          'SMOKE100' , 'DRNK3GE5' , 'MAXDRNKS' , 'FLSHTMY3' , 'HIVTST7' ,
                          'SEATBELT' , 'DRNKDRI2' , 'PDIABTS1' , 'HEATTBCO' , 'TRNSGNDR' ,
                          'LSATISFY' , 'EMTSUPRT' , 'SDLONELY' , 'SDHEMPLY' , 'FOODSTMP' ,
                          'SDHFOOD1' , 'SDHBILLS' , 'SDHUTILS' , 'SDHTRNSP' , 'SDHSTRE1' ,
                          'QSTVER' , 'QSTLANG' , 'HTIN4' , 'HTM4' , 'WTKG3' , 'DRNKANY6' ,
                          'EXRACT22', 'EXEROFT2', 'EXERHMM2', 'PREDIAB2', 'EXRACT12']) # 52
df23.shape

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

missing_columns

# 7-9, 77-99, 777-999, 7777-9999 olarak kodlanmış (bilmiyorum - cevap vemrek istemiyorum anlamındaki) gözlemler veri setinden çıkartıldı
cols = [col for col in df23.columns if col not in ["SEQNO", "SEXVAR", "STATE", "AGECAT"]]
for col in cols:
  if df23[col].max() < 10:
    df23 = df23[~df23[col].isin([7, 9])]
  elif df23[col].max() < 100:
    df23 = df23[~df23[col].isin([77, 99])]
  elif df23[col].max() < 1000:
    df23 = df23[~df23[col].isin([777, 999])]
  elif df23[col].max() < 10000:
    df23 = df23[~df23[col].isin([7777, 9999])]

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

df23.shape

#######################################################################
# Birim Dönüşümleri
#######################################################################

# Codebook referans alınarak kilo ve boy değişkenlerinin kg ve cm cinsine dönüştürmek için oluşturulan fonksiyonlar
def converting_height(height):
  if 0 <= height <= 8999:
    return ((height // 100) * 12 + (height % 100)) * 0.0254
  elif 9000 <= height < 10000:
    return (height - 9000) / 100

def converting_weight(weight):
  if 0 < weight < 8999:
    return weight * 0.4535924
  elif 9000 < weight < 10000:
    return weight - 9000

df23["HEIGHTCM"] = df23["HEIGHT3"].apply(converting_height)
df23["WEIGHTKG"] = df23["WEIGHT2"].apply(converting_weight)
df23 = df23.drop(columns=["WEIGHT2", "HEIGHT3"])

# EXRACTM_ değişkeni: Günlük ortalama aktivite süresi
def calculate_daily_minutes(row):
    freq = row["EXEROFT1"]
    duration = row["EXERHMM1"]
    if pd.isnull(freq) or pd.isnull(duration):
        return None

    if 101 <= freq <= 199:
        times_per_week = freq - 100
        daily_freq = times_per_week / 7
    elif 201 <= freq <= 299:
        times_per_month = freq - 200
        daily_freq = times_per_month / 30
    else:
        return None

    if 1 <= duration <= 759:
        minutes = duration
    elif 800 <= duration <= 899:
        minutes = (duration - 800) * 60
    else:
        return None
    return round(daily_freq * minutes, 2)

df23["EXRACTM"] = df23.apply(calculate_daily_minutes, axis=1)
df23 = df23.drop(columns=["EXEROFT1", "EXERHMM1"])

df23.shape

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

#######################################################################
# Missing Değerler
#######################################################################

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

# Missing değerlerin yerini doldurmak için oluşturulmuş fonksiyon
# Codebook referans alınmıştır.
def missing_values(dataframe, target_col, cond_col, cond_value, target_value):
    condition = (dataframe[cond_col] == cond_value) & (dataframe[target_col].isna())
    dataframe.loc[condition, target_col] = target_value

missing_values(df23, "BPMEDS1", "BPHIGH6", 2, 2)
missing_values(df23, "BPMEDS1", "BPHIGH6", 3, 2)
missing_values(df23, "BPMEDS1", "BPHIGH6", 4, 2)
missing_values(df23, "TOLDHI3", "CHOLCHK3", 1, 2)
missing_values(df23, "CHOLMED3", "CHOLCHK3", 1, 2)
df23 = df23.drop(columns=["CHOLCHK3"])
df23["ALCDAY4"].fillna(df23["ALCDAY4"].mean())

missing_values(df23, "AVEDRNK3", "ALCDAY4", 888, 88)
df23 = df23.drop(columns=["ALCDAY4"])
df23["SHINGLE2"].fillna(2, inplace=True)
missing_values(df23, "COVIDSM1", "COVIDPO1", 2, 2)
df23["RACE"].fillna(df23["RACE"].mode()[0], inplace=True)
df23["HEIGHTCM"].fillna(df23["HEIGHTCM"].mean(), inplace=True)
missing_values(df23, "EXRACTM", "EXERANY2", 2, 0)

# Orijinal veri setinde aslında 0 olan değerler 88 olarak kodlanmıştır.
# Bu değerler 0 olarak değiştirilmiştir.
df23[["PHYSHLTH", "MENTHLTH", "AVEDRNK3"]] = df23[["PHYSHLTH", "MENTHLTH", "AVEDRNK3"]].replace(88, 0)

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

# Referansı olmayan missing değerler veri setinden düşürüldü
df23.dropna(inplace=True, subset=['CHOLMED3', 'EXRACTM'])
df23.shape

df23.head()

missing_counts, missing_report, missing_percentages, missing_columns = report_missing(df23)
missing_report

df23.shape

#######################################################################
# Değişken İsimleri Güncellemesi
#######################################################################

# Değişken isimleri anlaşılır olacak şekilde değiştirilmiştir
new_cols = ["Sex", "GeneralHealth", "PhysicalHealth", "MentalHealth",
            "DoingExercise", "HighBloodPressure", "HighBloodPressureDrug",
            "HighColesterol", "HighColesterolDrug", "HadHeartAttack", "HadCoronaryHeartDisease",
            "HadStroke", "HadAsthma", "HadSkinCancer", "HadOtherCancer", "HadLungDisease",
            "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabate",
            "Deaf", "Blind", "DifficultyWalking", "SmokeUsing", "ECigaretteUsing",
            "AlcoholUsingAver", "HadFluVac", "HadPneumVac", "HadZonaVac", "HadCovid",
            "HadCovidSemp", "AgeCategory", "Race", "Height", "Weight", "DailyExerciseAver"]
len(new_cols)

df23.columns = new_cols

# Sayısal olarak görülen fakat kategorik olan değişkenler için derekli düzenlemeler yapılmıştır
# Kategorik değişkenler 'object' olarak değiştirilmiştir
YES_NO_QUESTIONS = {1: 'Yes', 2: 'No'}
unique_2_cols = [col for col in df23.columns if col != "Sex" and df23[col].nunique() == 2]
for col in unique_2_cols:
  df23[col] = df23[col].map(YES_NO_QUESTIONS)

SEX = {1: 'Male', 2: 'Female'}
df23["Sex"] = df23["Sex"].map(SEX)

GENERAL_HEALTH = {1: 'Excellent', 2: 'Very good', 3: "Good", 4: "Fair", 5: "Poor"}
df23["GeneralHealth"] = df23["GeneralHealth"].map(GENERAL_HEALTH)

BLOOD_PRESSURE = {1: "Yes", 2: "Yes, but female told only during pregnancy", 3: "No", 4: "Told borderline high or pre-hypertensive or elevated blood pressure"}
df23["HighBloodPressure"] = df23["HighBloodPressure"].map(BLOOD_PRESSURE)

DIABETE = {1: "Yes", 2: "Yes, but female told only during pregnancy", 3: "No", 4: "No, pre-diabetes or borderline diabetes"}
df23["HadDiabate"] = df23["HadDiabate"].map(DIABETE)

SMOKE = {1: "Every day", 2: "Some days", 3: "Not at all"}
df23["SmokeUsing"] = df23["SmokeUsing"].map(SMOKE)

ECIGARETTE = {1: "Not at all", 2: "Every day", 3: "Some days", 4: "Not at all"}
df23["ECigaretteUsing"] = df23["ECigaretteUsing"].map(ECIGARETTE)

RACE = {1: 'White only, Non-Hispanic',
        2: 'Black only, Non-Hispanic',
        3: "Other race only, Non-Hispanic",
        4: "Multiracial, Non-Hispanic",
        5: "Hispanic"}
df23["Race"] = df23["Race"].map(RACE)

AGE_CATEGORY = {1: '18 to 24', 2: '25 to 29', 3: "30 to 34", 4: "35 to 39", 5: "40 to 44",
                6: "45 to 49", 7: "50 to 54", 8: "55 to 59", 9: "60 to 64", 10: "65 to 69",
                11: "70 to 74", 12: "75 to 79", 13: "80 or older"}
df23["AgeCategory"] = df23["AgeCategory"].map(AGE_CATEGORY)

df23.head()

#######################################################################
# Aykırı Değerler
#######################################################################

df23.dtypes

df23.shape

# Kategorik ve sayısal değişkenlerin yakalanması
def grab_col_names(dataframe):
  # cat_cols, cat_but_car
  cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes=="O"]
  # num_cols
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
  num_cols = [col for col in num_cols if col not in cat_cols]

  print(f"Observation: {dataframe.shape[0]}")
  print(f"Variables: {dataframe.shape[1]}")
  print(f"cat_cols: {len(cat_cols)}")
  print(f"num_cols: {len(num_cols)}")
  return cat_cols, num_cols

cat_cols, num_cols = grab_col_names(df23)

num_cols

cat_cols

def num_summary(dataframe, numerical_col, plot=False):
  quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
  print(dataframe[numerical_col].describe(quantiles).T)

  if plot:
    plt.figure(figsize=(12, 20))  # Grafik boyutu burada ayarlanıyor
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.title(numerical_col)
    plt.show(block=True)

def cat_summary(dataframe, col_name, plot=False):
  print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                      "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
  print("##########################################")
  if plot:
    sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.show()

num_summary(df23, num_cols, plot=True)

# Boy değişkeni için aykırı değer analizi
df23[["Height"]].boxplot()
plt.title("Height")
plt.show()

# Kilo değişkeni için aykırı değer analizi
df23[["Weight"]].boxplot()
plt.title("Weight")
plt.show()

# Günlük ortalama egzersiz süresi değişkeni için aykırı değer analizi
df23[["DailyExerciseAver"]].boxplot()
plt.title("Daily Average Exercise")
plt.show()

# "Weight", "Height", ve "DailyExercisseAver" içerisinde fazlasıyla aykırı değer gözlemlenmiştir.
# Literatür araştırması ile veri setinin içerisinden bu değerler düşürülmüştür
df23 = df23[
    (df23['Weight'] >= 40) &
    (df23['Height'] >= 1.40) &
    (df23['Height'] <= 2.10) &
    (df23["DailyExerciseAver"] <= 300)
]
df23.shape

df23[["Weight"]].boxplot()
plt.title("Weight")
plt.show()

df23[["Height"]].boxplot()
plt.title("Height")
plt.show()

df23[["DailyExerciseAver"]].boxplot()
plt.title("Daily Average Exercise")
plt.show()

num_summary(df23, num_cols, plot=True)

df23.shape

for col in cat_cols:
    cat_summary(df23, col, plot=False)

#######################################################################
# Yeni Değişkenler
#######################################################################

df23["BMI_"] = df23["Weight"] / (df23["Height"] ** 2)
def bmi_kategori(bmi):
  if bmi < 18.5:
    return "Underweight"
  elif bmi < 25:
    return "Normal"
  elif bmi < 30:
    return "Overweight"
  elif bmi < 35:
    return "Obese (Class I)"
  elif bmi < 40:
    return "Obese (Class II)"
  else:
    return "Obese (Class I)"
df23["BMI_Category"] = df23["BMI_"].apply(bmi_kategori)

bmi_counts = df23["BMI_Category"].value_counts().sort_index()  # Kategorileri sıralı tutmak için sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=bmi_counts.index, y=bmi_counts.values, palette="Set2")

plt.title("BMI Kategorilerine Göre Kişi Sayısı", fontsize=14)
plt.xlabel("BMI Category")
plt.ylabel("Kişi Sayısı")
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Kronik Hastalık Skoru
chronic_cols = ["HadHeartAttack", "HadCoronaryHeartDisease", "HadStroke", "HadAsthma", "HadSkinCancer",
                "HadOtherCancer", "HadLungDisease", "HadDepressiveDisorder", "HadKidneyDisease",
                "HadArthritis", "HadDiabate"]
df23["Cronic_Count"] = df23[chronic_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

# Aşı Skoru
vaccine_cols = ["HadFluVac", "HadPneumVac", "HadZonaVac"]
df23["Vaccine_Score"] = df23[vaccine_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

# Sağlık Tabiki
track_cols = ["HighBloodPressure", "HighColesterol"]
df23["Health_Tracking"] = df23[track_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

# İlaç Tabiki
drug_cols = ["HighBloodPressureDrug", "HighColesterolDrug"]
df23["Drug_Tracking"] = df23[drug_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

# Yaşlılık Riski
senior_ages = ["65 to 69", "70 to 74", "75 to 79", "80 or older"]
# Yeni sütunu oluştur
df23["Old_Risk"] = df23["AgeCategory"].apply(lambda x: "Yes" if x in senior_ages else "No")

# Nikotin Skoru
def calculate_nicotin_score(smoke, ecig):
  smoke_score = 0
  ecig_score = 0

  # SmokeUsing için puan
  if smoke == "Not at all (right now)":
    smoke_score = 0
  elif smoke == "Some days":
    smoke_score = 1
  elif smoke == "Every day":
    smoke_score = 2

  # ECigaretteUsing için puan
  if ecig == "Not at all (right now)":
    ecig_score = 0
  elif ecig == "Some days":
    ecig_score = 1
  elif ecig == "Every day":
    ecig_score = 2

  # Toplam skoru döndür
  return smoke_score + ecig_score

# Skoru oluşturmak için apply kullanıyoruz
df23["Nicotin_Score"] = df23.apply(lambda row: calculate_nicotin_score(row["SmokeUsing"], row["ECigaretteUsing"]), axis=1)

# Risk Skoru
df23["Risk_Score"] = (df23["Cronic_Count"] + df23["Health_Tracking"] + df23["Nicotin_Score"] + (df23["BMI_"] > 35).astype(int) + (df23["Old_Risk"] == "Yes").astype(int))

def risk_cat(risk_level):
  if 0 <= risk_level <= 5:
    return "Risk Level 1"
  elif 6 <= risk_level <= 10:
    return "Risk Level 2"
  elif 11 <= risk_level <= 16:
    return "Risk Level 3"
df23["Risk_Level"] = df23["Risk_Score"].apply(risk_cat)

df23.shape

df23.to_csv("temizlenmis_cdc_23.csv", index=False)

#######################################################################
# Encoding İşlemleri
#######################################################################


def label_encoder(dataframe, binary_cols):
    labelencoder = LabelEncoder()
    for col in binary_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype="int64")
    return dataframe

binary_cols = [col for col in df23.columns if df23[col].dtype not in [int, float]
               and df23[col].nunique() == 2]
ohe_cols = [col for col in df23.columns if 30 >= df23[col].nunique() > 2]

df23 = label_encoder(df23, binary_cols)

df23 = one_hot_encoder(df23, ohe_cols)

#######################################################################
# Modelleme
#######################################################################

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix,  roc_curve, auc

df23.head()

y = df23["HadHeartAttack"]
X = df23.drop("HadHeartAttack", axis=1)

X.columns = X.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Sadece eğitim verisine SMOTE uygulanmıştır
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

lgbm_model = LGBMClassifier(random_state=17).fit(X_train_resampled, y_train_resampled)

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

y_pred = lgbm_model.predict(X_test)

def print_classification_results(y_true, y_pred):
    # Classification Report
    print("Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df.round(2))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

print_classification_results(y_test, y_pred)

best_params = {
    'learning_rate': 0.05,
    'max_depth': 7,
    'n_estimators': 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8}
lgbm_final = lgbm_model.set_params(**best_params, random_state=17).fit(X_train_resampled, y_train_resampled)
y_pred = lgbm_final.predict(X_test)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("Train Accuracy:", lgbm_final.score(X_train_resampled, y_train_resampled))
print("Test Accuracy :", lgbm_final.score(X_test, y_test))

print_classification_results(y_test, y_pred)

def plot_learning_curve(model, X, y, cv=5, scoring='roc_auc'):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Train Score", color="blue")
    plt.plot(train_sizes, test_scores_mean, label="Validation Score", color="orange")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_learning_curve(lgbm_final, X, y)

plot_learning_curve(lgbm_final, X, y, scoring='accuracy')

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X, num=len(X), save=False)

def plot_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc

plot_roc_curve(lgbm_final, X_test, y_test)

#######################################################################
# Model Testi
#######################################################################

random_index = np.random.randint(0, len(X_test))
random_user = X_test.iloc[[random_index]]
true_label = y_test.iloc[random_index]

prediction = lgbm_final.predict(random_user)[0]
proba = lgbm_final.predict_proba(random_user)[0][1]

print(f"Seçilen bireyin index numarası: {random_index}")
print("Gerçek sınıf:", "Kalp krizi riski VAR" if true_label == 1 else "Riski YOK")
print("Model tahmini:", "Kalp krizi riski VAR" if prediction == 1 else "Riski YOK")
print(f"Risk olasılığı (1 sınıfı): {proba:.2f}")

