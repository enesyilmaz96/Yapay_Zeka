#!/usr/bin/env python
# coding: utf-8

# # End-to-End Diabetes Machine Learning 
# 
# ## Pipeline II
# 

# In[1]:


# joblib kütüphanesi, model ve veri işlemlerini daha hızlı bir şekilde yapmak için kullanılır.
import joblib

# pandas, veri analizi ve manipülasyonu için yaygın olarak kullanılan bir kütüphanedir.
import pandas as pd

# LGBMClassifier, LightGBM adlı hafif ve hızlı bir gradient boosting çerçevesinin sınıflandırma modelini içerir.
from lightgbm import LGBMClassifier

# RandomForestClassifier, bir dizi karar ağacını eğiterek toplu olarak bir sınıflandırma modeli oluşturan bir sınıflandırıcıdır.
from sklearn.ensemble import RandomForestClassifier

# GradientBoostingClassifier, zayıf tahmincilerden güçlü bir tahminci oluşturan bir gradient boosting sınıflandırıcıdır.
from sklearn.ensemble import GradientBoostingClassifier

# VotingClassifier, birden çok sınıflandırıcıyı bir araya getirerek onların oylarıyla bir sınıflandırma modeli oluşturan bir sınıflandırıcıdır.
from sklearn.ensemble import VotingClassifier

# AdaBoostClassifier, zayıf sınıflandırıcıları birleştirerek güçlü bir sınıflandırıcı oluşturan bir sınıflandırıcıdır.
from sklearn.ensemble import AdaBoostClassifier

# LogisticRegression, lojistik regresyon modelini içeren bir sınıflandırıcıdır.
from sklearn.linear_model import LogisticRegression

# cross_validate, çeşitli ölçümleri kullanarak bir modelin performansını değerlendirmek için kullanılır.
from sklearn.model_selection import cross_validate

# GridSearchCV, bir modelin hiperparametrelerini belirlemek için kapsamlı bir arama stratejisi uygulayan bir sınıflandırıcıdır.
from sklearn.model_selection import GridSearchCV

# KNeighborsClassifier, k-en yakın komşular algoritmasını temel alan bir sınıflandırıcıdır.
from sklearn.neighbors import KNeighborsClassifier

# StandardScaler, veriyi standardize etmek (ortalama = 0, varyans = 1) için kullanılır.
from sklearn.preprocessing import StandardScaler

# SVC, destek vektör makinelerini temel alan bir sınıflandırıcıdır.
from sklearn.svm import SVC

# DecisionTreeClassifier, bir karar ağacı oluşturarak sınıflandırma modeli oluşturan bir sınıflandırıcıdır.
from sklearn.tree import DecisionTreeClassifier

# XGBClassifier, eXtreme Gradient Boosting (XGBoost) adlı bir hafif ve etkili gradient boosting kütüphanesinin sınıflandırma modelini içerir.
from xgboost import XGBClassifier


# # Helper Functions
# 
# utils veya helpers adında bir dosya oluşturarak buradaki fonksiyonları onunun içerisine atabilir ve sadece ilgili fonksiyonu import edebilirsiniz. Genellikle kodun içerisinde fazla yer kaplamaması için bu yöntem kullanılır ve sadece gerekli olan fonksiyon çağırılır. Aşağıdaki örneklerde olduğu gibi.

# In[2]:


# utils.py (İçerisindeki tüm fonksiyonları çeker)
# helpers.py (İçerisindeki tüm fonksiyonları çeker)
# from helpers import diabetes_data_prep (ilgili fonksiyonu çeker)


# Fakat biz burada bu işlemi yapmadan devam edeceğiz.

# # Data Preprocessing & Feature Engineering

# Bu fonksiyon, bir veri çerçevesinin sütunlarını analiz ederek kategorik değişkenleri, numerik değişkenleri, ve kategorik görünümlü kardinal (çok sayıda farklı değeri olan) değişkenleri belirlemek için kullanılır. İşte bu fonksiyonun açıklamalarla birlikte anlatımı:

# In[3]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
            Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
            numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
            kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
            Kategorik değişken listesi
        num_cols: list
            Numerik değişken listesi
        cat_but_car: list
            Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# Bu fonksiyon, **verilen bir sütundaki aykırı değerlerin alt ve üst limitlerini hesaplar**. Fonksiyonun dökümantasyonu, fonksiyonun kullanımı, aldığı parametreler ve döndürdüğü değerler hakkında bilgi verir.

# In[4]:


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Verilen sütun için aykırı değer limitlerini hesaplar.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Aykırı değer limitleri hesaplanacak veri çerçevesi.
    col_name : str
        Aykırı değer limitleri hesaplanacak sütun adı.
    q1 : float, optional
        İlk çeyrek (1. çeyrek) için yüzdelik dilim. Varsayılan değeri 0.25'tir.
    q3 : float, optional
        Üçüncü çeyrek (3. çeyrek) için yüzdelik dilim. Varsayılan değeri 0.75'tir.

    Returns
    -------
    low_limit : float
        Aykırı değerlerin alt limiti.
    up_limit : float
        Aykırı değerlerin üst limiti.

    Examples
    --------
    import pandas as pd
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    low_limit, up_limit = outlier_thresholds(df, 'col1')
    print(f'Lower Limit: {low_limit}, Upper Limit: {up_limit}')

    """
    # Sütundaki değerlerin ilk çeyrek (1. çeyrek) ve üçüncü çeyrek (3. çeyrek) yüzdelik dilimlerini hesapla
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    # İnterquartile range'i hesapla (IQR = Q3 - Q1)
    interquantile_range = quartile3 - quartile1

    # Aykırı değerlerin üst limitini hesapla (Q3 + 1.5 * IQR)
    up_limit = quartile3 + 1.5 * interquantile_range

    # Aykırı değerlerin alt limitini hesapla (Q1 - 1.5 * IQR)
    low_limit = quartile1 - 1.5 * interquantile_range

    # Hesaplanan limitleri döndür
    return low_limit, up_limit


# Bu fonksiyon, verilen bir sütundaki **aykırı değerleri belirlenen alt ve üst limitler arasına getirmek için kullanılır**. Fonksiyon, veri çerçevesini değiştirir, ancak herhangi bir değer döndürmez (**None**). Fonksiyonun dökümantasyonu, kullanımı, aldığı parametreler ve yaptığı iş hakkında bilgi verir.

# In[5]:


def replace_with_thresholds(dataframe, variable):
    """
    Verilen sütundaki aykırı değerleri belirtilen alt ve üst limitler arasına çeker.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Aykırı değerlerin düzenleneceği veri çerçevesi.
    variable : str
        Aykırı değerleri düzenlenecek sütun adı.

    Returns
    -------
    None

    Examples
    --------
    import pandas as pd
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    replace_with_thresholds(df, 'col1')
    print(df)

    """
    # Aykırı değerlerin alt ve üst limitlerini al
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    # Aykırı değerlerin alt limitinden küçük olanları alt limit ile değiştir
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

    # Aykırı değerlerin üst limitinden büyük olanları üst limit ile değiştir
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Bu fonksiyon, belirtilen **kategorik sütunları one-hot encoding ile dönüştürür ve yeni sütunlar ekleyerek** veri çerçevesini günceller. **drop_first** parametresi, dummy değişken tuzağını önlemek için ilk sütunu kaldırmak isteyip istemediğinizi belirler. Fonksiyonun dökümantasyonu, kullanımı, aldığı parametreler ve yaptığı iş hakkında bilgi verir.

# In[6]:


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Verilen veri çerçevesindeki belirli kategorik sütunları one-hot encoding ile dönüştürür.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        One-hot encoding uygulanacak veri çerçevesi.
    categorical_cols : list
        One-hot encoding uygulanacak kategorik sütun adlarını içeren liste.
    drop_first : bool, optional
        Dummy değişken tuzağını (dummy variable trap) önlemek için ilk sütunu kaldırma. Varsayılan değeri False'tur.

    Returns
    -------
    dataframe : pandas.DataFrame
        One-hot encoding uygulanan veri çerçevesi.

    Examples
    --------
    import pandas as pd
    df = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B']})
    df_encoded = one_hot_encoder(df, ['Category'])
    print(df_encoded)

    """
    # pandas'ın get_dummies fonksiyonu ile one-hot encoding uygula
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

    # one-hot encoding uygulanan veri çerçevesini döndür
    return dataframe


# Bu fonksiyon, diyabet **veri setini hazırlamak için bir dizi ön işleme adımı uygular**. Bu adımlar arasında **sütun isimlerini büyük harfe çevirme, yeni kategorik sütunlar oluşturma, kategorik sütunları one-hot encoding ile dönüştürme, aykırı değerleri belirli limitler arasına çekme ve numerik sütunları standartlaştırma bulunmaktadır**. Fonksiyon, bağımsız değişkenleri (X) ve bağımlı değişkeni (y) içeren bir çift döndürür.

# In[7]:


def diabetes_data_prep(dataframe):
    """
    Diyabet veri setini hazırlar.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Diyabet veri setini içeren veri çerçevesi.

    Returns
    -------
    X : pandas.DataFrame
        Bağımsız değişkenleri içeren veri çerçevesi.
    y : pandas.Series
        Bağımlı değişkeni içeren seri.

    Examples
    --------
    import pandas as pd
    df = pd.DataFrame({'age': [25, 30, 35], 'glucose': [80, 100, 120], 'outcome': [0, 1, 0]})
    X, y = diabetes_data_prep(df)
    print(X)
    print(y)

    """
    # Sütun isimlerini büyük harfe çevir
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Glucose sütununu kullanarak yeni bir kategorik sütun oluştur
    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

    # Age sütununu kullanarak yeni bir kategorik sütun oluştur
    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    # BMI sütununu kullanarak yeni bir kategorik sütun oluştur
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                        labels=["underweight", "healty", "overweight", "obese"])

    # BloodPressure sütununu kullanarak yeni bir kategorik sütun oluştur
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                            labels=["normal", "hs1", "hs2"])

    # Kategorik, numerik ve kategorik görünümlü kardinal sütunları al
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    # "OUTCOME" sütununu içermeyen kategorik sütunları seç
    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

    # One-hot encoding uygula
    df = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    # Kategorik, numerik ve kategorik görünümlü kardinal sütunları tekrar al
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    # "INSULIN" sütununda aykırı değerleri belirlenen limitler arasına çek
    replace_with_thresholds(df, "INSULIN")

    # Numerik sütunları standartlaştır
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    # Bağımlı ve bağımsız değişkenleri ayır
    y = df["OUTCOME"]
    X = df.drop(["OUTCOME"], axis=1)

    return X, y


# # Base Models

# Bu fonksiyon, **belirtilen skorlama metriği altında çeşitli temel sınıflandırma modellerini eğitir ve performanslarını ekrana yazdırır**. scoring parametresi, hangi skorlama metriğinin kullanılacağını belirler (varsayılan olarak "roc_auc"). Fonksiyonun dökümantasyonu, kullanımı, aldığı parametreler ve yaptığı iş hakkında bilgi verir.

# In[8]:


def base_models(X, y, scoring="roc_auc"):
    """
    Verilen veri setindeki temel (base) modelleri eğitir ve performanslarını değerlendirir.

    Parameters
    ----------
    X : pandas.DataFrame
        Bağımsız değişkenleri içeren veri çerçevesi.
    y : pandas.Series
        Bağımlı değişkeni içeren seri.
    scoring : str, optional
        Performans değerlendirmesi için kullanılacak skorlama metriği. Varsayılan olarak "roc_auc"tur.

    Returns
    -------
    None

    Examples
    --------
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    base_models(X_train_scaled, y_train)

    """
    print("Base Models....")
    
    # Kullanılacak sınıfları ve modelleri tanımla
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    # Her bir modeli eğit ve performansını değerlendir
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# # Hyperparameter Optimization

# Başlangıçta bahsettiğim uygulama burada da gerçekleştirilebilir. Aşağıda gördüğünüz gibi her sınıflandırma modeli için parametre değerleri verilmiştir.
# Buradaki bilgiler config adında bir dosyaya kaydedilerek. Kullanılması gerektiğinde çağırılabilir. Gereken durumlarda içerisindeki değerler değiştirilebilir.

# In[9]:


# config.py


# Bu fonksiyon, **belirtilen sınıflandırma modelleri için hiperparametre optimizasyonu yapar**. Her bir model için **GridSearchCV kullanarak en iyi hiperparametreleri bulur ve en iyi modeli seçer**. Fonksiyon, en iyi modellerin bir sözlüğünü döndürür. Dökümantasyon, kullanım, aldığı parametreler ve yaptığı iş hakkında bilgi verir.

# In[10]:


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    """
    Verilen veri setindeki sınıflandırma modelleri için hiperparametre optimizasyonu gerçekleştirir.

    Parameters
    ----------
    X : pandas.DataFrame
        Bağımsız değişkenleri içeren veri çerçevesi.
    y : pandas.Series
        Bağımlı değişkeni içeren seri.
    cv : int, optional
        Çapraz doğrulama kat sayısı. Varsayılan olarak 3'tür.
    scoring : str, optional
        Performans değerlendirmesi için kullanılacak skorlama metriği. Varsayılan olarak "roc_auc"tur.

    Returns
    -------
    best_models : dict
        Hiperparametre optimizasyonu sonucunda elde edilen en iyi modellerin sözlüğü.

    Examples
    --------
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    best_models = hyperparameter_optimization(X_train_scaled, y_train)

    """
    print("Hyperparameter Optimization....")
    best_models = {}
    
    # Her bir sınıflandırma modeli için hiperparametre optimizasyonu gerçekleştir
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        
        # Modelin başlangıç performansını değerlendir
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        # GridSearchCV ile hiperparametre optimizasyonu yap
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        # Hiperparametre optimizasyonu sonrası performansı değerlendir
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        
        # En iyi modeli sözlüğe ekle
        best_models[name] = final_model
    
    return best_models


# # Stacking & Ensemble Learning

# Bu fonksiyon, belirtilen modelleri kullanarak bir "**Voting Classifier**" oluşturur ve bu **ensemble modelin performansını değerlendirir**. Fonksiyon, oluşturulan modeli döndürür. Dökümantasyon, kullanım, aldığı parametreler ve yaptığı iş hakkında bilgi verir.

# In[11]:


def voting_classifier(best_models, X, y):
    """
    Belirtilen sınıflandırma modellerini kullanarak bir "Voting Classifier" oluşturur ve performansını değerlendirir.

    Parameters
    ----------
    best_models : dict
        Hiperparametre optimizasyonu sonucunda elde edilen en iyi modellerin sözlüğü.
    X : pandas.DataFrame
        Bağımsız değişkenleri içeren veri çerçevesi.
    y : pandas.Series
        Bağımlı değişkeni içeren seri.

    Returns
    -------
    voting_clf : sklearn.ensemble.VotingClassifier
        Oluşturulan "Voting Classifier" modeli.

    Examples
    --------
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    best_models = hyperparameter_optimization(X_train_scaled, y_train)
    voting_clf = voting_classifier(best_models, X_train_scaled, y_train)

    """
    print("Voting Classifier...")
    
    # Seçilen en iyi modelleri kullanarak bir Voting Classifier oluştur
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    
    # Oluşturulan modelin performansını değerlendir
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    
    return voting_clf


# # Pipeline Main Function

# Bu kod, belirtilen bir diyabet veri seti üzerinde çalışan bir dizi işlemi içerir. main fonksiyonu, veriyi yükler, hazırlar, temel modelleri eğitir, hiperparametre optimizasyonu yapar, bir "**Voting Classifier**" oluşturur ve en iyi modeli kaydeder. if __name__ == "__main__": kısmı, bu script'in başka bir script veya modül tarafından çağrıldığında çalışmasını sağlar.

# In[12]:


def main():
    """
    Ana işlem fonksiyonu. Veri setini yükler, veriyi hazırlar, temel modelleri eğitir, hiperparametre optimizasyonu yapar,
    bir "Voting Classifier" oluşturur ve en iyi modeli kaydeder.

    Returns
    -------
    voting_clf : sklearn.ensemble.VotingClassifier
        Oluşturulan "Voting Classifier" modeli.

    Examples
    --------
    main()

    """
    df = pd.read_csv("datasets/diabetes.csv")  # Veri setini CSV dosyasından yükle
    X, y = diabetes_data_prep(df)  # Veriyi hazırla

    # Temel modelleri eğit ve performanslarını değerlendir
    base_models(X, y)

    # Hiperparametre optimizasyonu yaparak en iyi modelleri bul
    best_models = hyperparameter_optimization(X, y)

    # En iyi modelleri kullanarak bir Voting Classifier oluştur
    voting_clf = voting_classifier(best_models, X, y)

    # Oluşturulan modeli dosyaya kaydet
    joblib.dump(voting_clf, "voting_clf.pkl")

    return voting_clf


# Bu ifade, Python script'inin çalıştırılmasını kontrol etmek için kullanılır. 
# __name__ özel bir eğişken olup, bir Python script'inin modül olarak kullanılıp
# kullanılmadığını belirler. Eğer bir script doğrudan çalıştırılıyorsa (yani
# ana program olarak çalışıyorsa), __name__ değeri "__main__" olur.
# 

# In[13]:


if __name__ == "__main__":
    # Bu şekilde kullanımın avantajı main fonksiyonunu başlatmadan önce istediğimiz
    # Herhangi bir kodu çalıştırabiliriz. Örneğin bir print ekleyelim;
    print("İşlem başladı")
    main()

