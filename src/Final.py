import pandas as pd
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import inspect
import seaborn as sns
from autorank import autorank, plot_stats
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sensible = []
privilege = []
unprivilege = []
df_original = None
methode = []

def check_if_numeric(dataframe):
    try:
        dataframe = dataframe.apply(pd.to_numeric, errors='raise')
        print("Toutes les données sont des nombres.")
    except ValueError as e:
        print("Certaines données ne sont pas des nombres.")
        print(e)

def takeDataframe(name, encoding):
    columns_adult = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
    'hours-per-week', 'native-country', 'income']

    # Recuperation des fichier csv en dataframe
    dossier = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset'))
    chemin = os.path.join(dossier, name)
    if((name == 'compas-scores-two-years.csv') or name == 'adult.data.txt' or name == 'lawschool.csv' or name == 'german_credit_risk.csv' ):
        dataframe = pd.read_csv(chemin,sep=',', encoding=encoding)
    elif name == 'default_credit.csv':
        dataframe = pd.read_csv(chemin, skiprows=1)
        dataframe.rename(columns={'default payment next month': 'yN'}, inplace=True)
    else:
        dataframe = pd.read_csv(chemin,sep=';', encoding=encoding)
    
    if name == 'adult.data.txt':
        dataframe.columns = columns_adult

    if( name == 'bank-full.csv'):
        month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        dataframe['month_numeric'] = dataframe['month'].map(month_mapping)
        dataframe.drop(columns=['month'],inplace= True)
    return dataframe

def numerique(dataframe) :
    # Numerise student-por
    dataframeN = dataframe
    age_mapping = {15: 1, 16: 1, 17: 1, 18: 1, 19: 0, 20: 0, 21: 0, 22: 0}
    dataframeN = dataframeN.assign(ageN = dataframe['age'].map(age_mapping).fillna(0).astype(int))
    school_mapping = {"GP":0, "MS":1}
    dataframeN = dataframeN.assign(schoolN = dataframe['school'].map(school_mapping))
    sex_mapping = {"F":0, "M":1}
    dataframeN = dataframeN.assign(sexN = dataframe['sex'].map(sex_mapping))
    address_mapping = {"U":0,"R":1}
    dataframeN = dataframeN.assign(addressN = dataframe['address'].map(address_mapping))
    famsize_mapping = {"LE3":0, "GT3":1}
    dataframeN = dataframeN.assign(famsizeN = dataframe['famsize'].map(famsize_mapping))
    Pstatus_mapping = {"T":0, "A":1}
    dataframeN = dataframeN.assign(PstatusN = dataframe['Pstatus'].map(Pstatus_mapping))
    job_mapping = {"teacher":0, "health":1, "services":2, "at_home":3, "other":4}
    dataframeN = dataframeN.assign(MjobN = dataframe['Mjob'].map(job_mapping))
    dataframeN = dataframeN.assign(FjobN = dataframe['Fjob'].map(job_mapping))
    reason_mapping = {"home":0, "reputation":1, "course":2, "other":3}
    dataframeN = dataframeN.assign(reasonN = dataframe['reason'].map(reason_mapping))
    guard_mapping = {"mother":0, "father":1, "other":2}
    dataframeN = dataframeN.assign(guardianN = dataframe['guardian'].map(guard_mapping))
    yesno_mapping = {"no":0, "yes":1}
    dataframeN = dataframeN.assign(schoolsupN = dataframe['schoolsup'].map(yesno_mapping))
    dataframeN = dataframeN.assign(famsupN = dataframe['famsup'].map(yesno_mapping))
    dataframeN = dataframeN.assign(paidN = dataframe['paid'].map(yesno_mapping))
    dataframeN = dataframeN.assign(activitiesN = dataframe['activities'].map(yesno_mapping))
    dataframeN = dataframeN.assign(nurseryN = dataframe['nursery'].map(yesno_mapping))
    dataframeN = dataframeN.assign(higherN = dataframe['higher'].map(yesno_mapping))
    dataframeN = dataframeN.assign(internetN = dataframe['internet'].map(yesno_mapping))
    dataframeN = dataframeN.assign(romanticN = dataframe['romantic'].map(yesno_mapping))
    value_mapping = {str(i): i for i in range(21)}
    #dataframeN = dataframeN.assign(G1N = dataframe['G1'].map(value_mapping))
    #dataframeN = dataframeN.assign(G2N = dataframe['G2'].map(value_mapping))
    dataframeN = dataframeN.assign(yN=dataframe['G3'].apply(lambda x: 0 if x <= 9 else 1))


    dataframeN.drop(columns=[
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
        'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
        'higher', 'internet', 'romantic', 'G3'
    ], inplace=True)

    return dataframeN
def numeric(dataframe) :
    # Numerise bank-full
    dataframeN = dataframe
    job_mapping = {'student':0,'unemployed':1, 'unknown':2, 'housemaid':3, 'blue-collar':4, 'services':5, 'retired':6, 'self-employed':7, 'technician':8 ,'management':9,'entrepreneur':10,'admin.':11 }
    dataframeN = dataframeN.assign(jobN = dataframe['job'].map(job_mapping))
    marital_mapping = {'divorced':0,'single':1,'married':2}
    dataframeN = dataframeN.assign(maritalN = dataframe['marital'].map(marital_mapping))
    education_mapping = {'primary':0,'unknown':1,'secondary':2,'tertiary':3}
    dataframeN = dataframeN.assign(educationN = dataframe['education'].map(education_mapping))
    yesno_mapping = {'yes': 1,'no':0}
    dataframeN = dataframeN.assign(defaultN = dataframe['default'].map(yesno_mapping))
    dataframeN = dataframeN.assign(housingN = dataframe['housing'].map(yesno_mapping))
    dataframeN = dataframeN.assign(loanN = dataframe['loan'].map(yesno_mapping))
    dataframeN = dataframeN.assign(yN = dataframe['y'].map(yesno_mapping))
    contact_mapping = {'unknown':0,'telephone':1,'cellular':2}
    dataframeN = dataframeN.assign(contactN = dataframe['contact'].map(contact_mapping))
    #poutcome_mapping = {'unknown':0,'failure':1,'other':2,'success':3}
    #dataframeN = dataframeN.assign(poutcomeN = dataframe['job'].map(poutcome_mapping))

    dataframeN.drop(columns=['job','marital','education','default','housing','loan','contact','poutcome','y'],inplace= True)

    return dataframeN
def numerique_adult(dataframe):
    dataframeN = dataframe
    
    dataframe['workclass'] = dataframe['workclass'].str.strip()
    workclass_mapping = {
        'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2,
        'Federal-gov': 3, 'Local-gov': 4, 'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7, '?' : 8
    }
    dataframeN = dataframeN.assign(workclassN = dataframe['workclass'].map(workclass_mapping))

    dataframe['education'] = dataframe['education'].str.strip()
    education_mapping = {
        'Bachelors': 0, 'Masters': 1, 'Doctorate': 2, 'Assoc-acdm': 3, 'Assoc-voc': 4,
        '11th': 5, 'HS-grad': 6, 'Prof-school': 7, 'Some-college': 8, '9th': 9, '7th-8th': 10,
        '12th': 11, '1st-4th': 12, '5th-6th': 13, '10th': 14, 'Preschool': 15
    }
    dataframeN = dataframeN.assign(educationN = dataframe['education'].map(education_mapping))

    dataframe['marital-status'] = dataframe['marital-status'].str.strip()
    marital_mapping = {'Never-married': 0, 'Married-civ-spouse': 1, 'Divorced': 2, 'Separated': 3,
                       'Married-spouse-absent': 4, 'Married-AF-spouse': 5, 'Widowed':6}
    dataframeN = dataframeN.assign(marital_statusN = dataframe['marital-status'].map(marital_mapping))

    dataframe['occupation'] = dataframe['occupation'].str.strip()
    occupation_mapping = {
        'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4, 'Prof-specialty': 5,
        'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9, 'Transport-moving': 10,
        'Priv-house-serv': 11, 'Protective-serv': 12, 'Armed-Forces': 13, "?" : 14
    }
    dataframeN = dataframeN.assign(occupationN = dataframe['occupation'].map(occupation_mapping))

    dataframe['relationship'] = dataframe['relationship'].str.strip()
    relationship_mapping = {
        'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 5
    }
    dataframeN = dataframeN.assign(relationshipN = dataframe['relationship'].map(relationship_mapping))

    dataframe['race'] = dataframe['race'].str.strip()
    race_mapping = {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4}
    dataframeN = dataframeN.assign(raceN = dataframe['race'].map(race_mapping))

    dataframe['sex'] = dataframe['sex'].str.strip()
    sex_mapping = {'Male': 0, 'Female': 1}
    dataframeN = dataframeN.assign(sexN = dataframe['sex'].map(sex_mapping))

    dataframe['native-country'] = dataframe['native-country'].str.strip()
    country_mapping = {
    'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5,
    'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10, 'China': 11,
    'Cuba': 12, 'Iran': 13, 'Honduras': 14, 'Philippines': 15, 'Italy': 16, 'Poland': 17, 'Jamaica': 18,
    'Vietnam': 19, 'Mexico': 20, 'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24,
    'Laos': 25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31,
    'Nicaragua': 32, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, '?':41,
    'El-Salvador': 37, 'Peru': 38, 'Trinadad&Tobago': 39, 'Hong': 40, 'Holand-Netherlands': 36
}
    dataframeN = dataframeN.assign(native_countryN = dataframe['native-country'].map(country_mapping))

    yesno_mapping = {' <=50K': 0, ' >50K': 1}
    dataframeN = dataframeN.assign(yN = dataframe['income'].map(yesno_mapping))

    dataframeN.drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                             'native-country', 'income'], inplace=True)

    return dataframeN
def numerique_compas(dataframe):
    dataframeN = dataframe
    
    race_mapping = {'Caucasian': 0, 'African-American': 1, 'Hispanic': 2, 'Other': 3}
    dataframeN = dataframeN.assign(raceN = dataframe['race'].map(race_mapping))

    sex_mapping = {"Male": 1, "Female": 0}
    dataframeN['sex'] = dataframe['sex'].map(sex_mapping)

    dataframeN = dataframeN.assign(priors_countN = dataframe['priors_count'].apply(lambda x: min(x, 10)))

    dataframeN = dataframeN.assign(yN = dataframe['two_year_recid'].map({0: 0, 1: 1}))

    columns_to_drop = [
        'juv_fel_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_charge_desc',
        'is_recid', 'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 
        'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid', 'is_violent_recid', 
        'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'screening_date', 
        'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'out_custody', 
        'start', 'end', 'event'
    ]
    
    columns_to_drop = [col for col in columns_to_drop if col in dataframe.columns]
    
    dataframeN = dataframe.drop(columns=columns_to_drop, errors='ignore') 

    return dataframeN
def numerique_german(dataframe):
    dataframeN = dataframe.copy()
    
    sex_mapping = {"female": 0, "male": 1}
    dataframeN = dataframeN.assign(sexN=dataframe['Sex'].map(sex_mapping))

    housing_mapping = {'own': 0, 'free': 1, 'rent': 2}
    dataframeN = dataframeN.assign(housingN=dataframe['Housing'].map(housing_mapping))

    saving_accounts_mapping = {'Unknown' : 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
    dataframeN = dataframeN.assign(saving_accountsN=dataframe['Saving accounts'].map(saving_accounts_mapping))

    checking_account_mapping = {'Unknown': 0, 'little': 1, 'moderate': 2, 'rich': 3}
    dataframeN = dataframeN.assign(checking_accountN=dataframe['Checking account'].map(checking_account_mapping))

    dataframeN['Credit amount'] = pd.to_numeric(dataframe['Credit amount'], errors='coerce')
    dataframeN['Duration'] = pd.to_numeric(dataframe['Duration'], errors='coerce')

    purpose_mapping = {
        'car': 0, 'business': 1, 'education': 2, 'radio/TV': 3, 
        'furniture/equipment': 4, 'domestic appliances': 5, 'repairs': 6, 'vacation/others': 7
    }
    dataframeN = dataframeN.assign(purposeN=dataframe['Purpose'].map(purpose_mapping))

    risk_mapping = {'good': 1, 'bad': 0}
    dataframeN = dataframeN.assign(yN=dataframe['Risk'].map(risk_mapping))

    dataframeN['AgeN'] = (dataframe['Age'] >= 25).astype(int)

    dataframeN.drop(columns=['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk', 'Age'], inplace=True)

    return dataframeN

def metric_privilege(dataframe):
    #Creation des objects nous permettant de recuper les metrics voulues
    metric = []
    for x in range(len(sensible)) :
        dataset = BinaryLabelDataset(df=dataframe, label_names=['yN'], protected_attribute_names=[sensible[x]])
        dataset_original = BinaryLabelDataset(df=df_original, label_names=['yN'], protected_attribute_names=[sensible[x]])
        unp_group = [{sensible[x]: v} for v in unprivilege[x]]
        p_group = [{sensible[x]: v} for v in privilege[x]]
        metric.append(ClassificationMetric(dataset_original, dataset,unprivileged_groups=unp_group,privileged_groups=p_group))
    return metric

def test(x_train, x_test, y_train, y_test) :
    global df_original
    df_original = pd.DataFrame(x_test)
    df_original ['yN'] = y_test

    # Normalise data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Normalise data using Min-Max Scaling
    scaler2 = MinMaxScaler()
    x_train_scaled2 = scaler2.fit_transform(x_train)
    x_test_scaled2 = scaler2.transform(x_test)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(x_train_scaled2, y_train)
    y_pred = nb_model.predict(x_test_scaled2)
    df_Multinnomina_bayes = pd.DataFrame(x_test)
    df_Multinnomina_bayes['yN'] = y_pred
    metric_bayes = metric_privilege(df_Multinnomina_bayes)

    # Logistic Regression
    regression_model = LogisticRegression()
    regression_model.fit(x_train_scaled, y_train)
    y_pred_regression = regression_model.predict(x_test_scaled)
    df_Reg = pd.DataFrame(x_test)
    df_Reg['yN'] = y_pred_regression
    metric_reg = metric_privilege(df_Reg)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=20)
    knn_model.fit(x_train_scaled, y_train)
    y_pred_knn = knn_model.predict(x_test_scaled)
    df_knn = pd.DataFrame(x_test)
    df_knn['yN'] = y_pred_knn
    metric_knn = metric_privilege(df_knn)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(x_train_scaled, y_train)
    y_pred_rf=rf_model.predict(x_test_scaled)
    df_rf = pd.DataFrame(x_test)
    df_rf['yN'] = y_pred_rf
    metric_rf = metric_privilege(df_rf)

    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=1)
    nn_model.fit(x_train_scaled, y_train)
    y_pred_nn = nn_model.predict(x_test_scaled)
    df_nn = pd.DataFrame(x_test)
    df_nn['yN'] = y_pred_nn
    metric_nn = metric_privilege(df_nn)

    # SVM
    svm_model = SVC()
    svm_model.fit(x_train, y_train)
    y_pred_svm = svm_model.predict(x_test)
    df_svm = pd.DataFrame(x_test)
    df_svm['yN'] = y_pred_svm
    metric_svm = metric_privilege(df_svm)

    resultat = [metric_bayes,metric_reg,metric_knn,metric_rf,metric_nn,metric_svm]
    return resultat

def test_hyper_parametre(x_train, x_test, y_train, y_test):
    global df_original
    df_original = pd.DataFrame(x_test)
    df_original ['yN'] = y_test

    # Normalise data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Normalise data using Min-Max Scaling
    scaler2 = MinMaxScaler()
    x_train_scaled2 = scaler2.fit_transform(x_train)
    x_test_scaled2 = scaler2.transform(x_test)

    # Naive Bayes
    alphas =[0.1, 1, 10]
    resultat_NB =[]
    for alpha in alphas:
        nb_model = MultinomialNB(alpha=alpha)
        nb_model.fit(x_train_scaled2, y_train)
        y_pred = nb_model.predict(x_test_scaled2)
        df_Multinnomina_bayes = pd.DataFrame(x_test)
        df_Multinnomina_bayes['yN'] = y_pred
        metric_bayes = metric_privilege(df_Multinnomina_bayes)
        resultat_NB.append(metric_bayes)


    # KNN
    neighbors=[5,10,20]
    resultat_KNN =[]
    for neighbor in neighbors:
        knn_model = KNeighborsClassifier(n_neighbors=neighbor)
        knn_model.fit(x_train_scaled, y_train)
        y_pred_knn = knn_model.predict(x_test_scaled)
        df_knn = pd.DataFrame(x_test)
        df_knn['yN'] = y_pred_knn
        metric_knn = metric_privilege(df_knn)
        resultat_KNN.append(metric_knn)

    # Random Forest
    estimators=[50,100,300]
    resultat_RF=[]
    for estimator in estimators:
        rf_model = RandomForestClassifier(n_estimators=estimator)
        rf_model.fit(x_train_scaled, y_train)
        y_pred_rf=rf_model.predict(x_test_scaled)
        df_rf = pd.DataFrame(x_test)
        df_rf['yN'] = y_pred_rf
        metric_rf = metric_privilege(df_rf)
        resultat_RF.append(metric_rf)

    # Neural Network
    layers =[(100,),(50, 50) ,(100, 50, 30)]
    resultat_NN=[]
    for layer in layers:
        nn_model = MLPClassifier(hidden_layer_sizes=layer, max_iter=1000, random_state=1)
        nn_model.fit(x_train_scaled, y_train)
        y_pred_nn = nn_model.predict(x_test_scaled)
        df_nn = pd.DataFrame(x_test)
        df_nn['yN'] = y_pred_nn
        metric_nn = metric_privilege(df_nn)
        resultat_NN.append(metric_nn)

    # SVM
    cs =[0.1, 1, 10]
    resultat_SVM=[]
    for c in cs:
        svm_model = SVC(C=c)
        svm_model.fit(x_train, y_train)
        y_pred_svm = svm_model.predict(x_test)
        df_svm = pd.DataFrame(x_test)
        df_svm['yN'] = y_pred_svm
        metric_svm = metric_privilege(df_svm)
        resultat_SVM.append(metric_svm)
    
    resultat=[resultat_NB,resultat_KNN,resultat_RF,resultat_NN,resultat_SVM]
    return resultat

def multiple(dataframe, n, bool):
    # lance les test n fois
    y = dataframe['yN']
    x = dataframe.drop(columns=['yN'])

    for column in x.columns:
        x[column] = x[column].astype('category').cat.codes

    kf = KFold(n_splits=n, shuffle=True, random_state=42)

    memoire = [0] * n
    m = 0
    for fold_idx, (train_index, test_index) in enumerate(kf.split(x)):
        print(f"Fold {fold_idx + 1}/{n}")  # Debug : afficher le fold en cours
        tmp = []
        # Séparer les données selon les indices du split
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if bool :
            tmp = (test(x_train, x_test, y_train, y_test))
        else :
            tmp = (test_hyper_parametre(x_train, x_test, y_train, y_test))
        memoire[m] = tmp
        m = m + 1
    return memoire

def moyenne(liste):
    taille = len(liste)
    nb_m = len(liste[0])
    nb_o = len(liste[0][0])
    nb_stat = 7

    # Initialisation des structures de données
    somme = [[[0.0 for _ in range(nb_stat)] for _ in range(nb_o)] for _ in range(nb_m)]
    valeurs = [[[[] for _ in range(nb_stat)] for _ in range(nb_o)] for _ in range(nb_m)]

    # Parcours de tous les résultats
    for x in liste:
        for m, y in enumerate(x):
            for o, z in enumerate(y):
                stats = [
                    z.disparate_impact(),
                    z.equal_opportunity_difference(),
                    z.accuracy(),
                    z.error_rate_difference(),
                    (z.true_negative_rate() + z.true_positive_rate()) / 2,
                    z.generalized_entropy_index(),
                    0.0  # F1-score sera calculé à part
                ]

                precision = z.precision()
                recall = z.recall()
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                stats[6] = f1

                for i in range(nb_stat):
                    somme[m][o][i] += stats[i]
                    valeurs[m][o][i].append(stats[i])

    # Moyenne
    moyenne = [[[somme[m][o][i] / taille for i in range(nb_stat)] for o in range(nb_o)] for m in range(nb_m)]

    # Variance classique
    ecart_type = [[[0.0 for _ in range(nb_stat)] for _ in range(nb_o)] for _ in range(nb_m)]
    for m in range(nb_m):
        for o in range(nb_o):
            for i in range(nb_stat):
                variance = sum((val - moyenne[m][o][i]) ** 2 for val in valeurs[m][o][i]) / taille
                ecart_type[m][o][i] = math.sqrt(variance)

    return moyenne, ecart_type

def plot_bar(test_name, x, y, stdev, bidule, name):
    plt.figure()
    plt.bar(x, y, yerr=stdev, width=0.4)
    plt.xlabel("Methode")
    plt.ylabel(test_name)
    plt.title(name +" " + test_name)
    plt.savefig(f"./Graph/{name}_{bidule}_{test_name}.png")
    plt.close()

def plot_bar_parametre (test_name, x, y, stdev, bidule, name, methode):
    plt.figure()
    plt.bar(x, y, yerr=stdev, width=0.4)
    plt.xlabel(methode)
    plt.ylabel(test_name)
    plt.title(methode + test_name)
    plt.savefig(f"./Graph/{name}_{bidule}_{test_name}_{methode}.png")
    plt.close()

def calcul(test,name):
    result, stdev = moyenne(test)

    if not os.path.exists("./Graph"):
        print("Création Graph")
        os.makedirs("./Graph")

    # Tableau avec les resultats
    methode = (["NB", "LR", "KNN", "RF", "NN", "SVM"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index', 'F1 Score']


    print("Moyenne :")
    res_final=[]
    res_final.append(columns)
    for s in range(len(sensible)):
        for m in range(len(methode)):
            rows = []
            row_name = f"{methode[m]}_{sensible[s]}"
            rows.append(row_name)
            rows.extend(result[m][s])
            res_final.append(rows)

    for x in range(len(res_final)):
        print(res_final[x])

    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result[i][s][m] for i in range(len(methode))]
            standev = [stdev[i][s][m] for i in range(len(methode))]
            plot_bar(columns[m+1],methode,y,standev,sensible[s],name)

def calcul_hyperparametre(test,name):
    tableau_NB = [sous_tableau[0] for sous_tableau in test]
    result_NB, stdev_NB = moyenne(tableau_NB)

    methode = (["0.1", "1", "10"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index', 'F1 Score']


    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_NB[i][s][m] for i in range(len(methode))]
            standev = [stdev_NB[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","Naive_Bayes")

            tableau_NB = [sous_tableau[0] for sous_tableau in test]
    
    tableau_KNN = [sous_tableau[1] for sous_tableau in test]        
    result_KNN, stdev_KNN = moyenne(tableau_KNN)

    methode = (["5", "10", "20"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index', 'F1 Score']


    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_KNN[i][s][m] for i in range(len(methode))]
            standev = [stdev_KNN[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","k_neighbors")

    tableau_RF = [sous_tableau[2] for sous_tableau in test]        
    result_RF, stdev_RF = moyenne(tableau_RF)

    methode = (["50", "100", "300"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index', 'F1 Score']


    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_RF[i][s][m] for i in range(len(methode))]
            standev = [stdev_RF[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","Random_Forests")

    tableau_NN = [sous_tableau[3] for sous_tableau in test]        
    result_NN, stdev_NN = moyenne(tableau_NN)

    methode = (["(100,)","(50, 50)" ,"(100, 50, 30)"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index', 'F1 Score']


    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_NN[i][s][m] for i in range(len(methode))]
            standev = [stdev_NN[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","Neural_networks")

    tableau_SVM = [sous_tableau[4] for sous_tableau in test]        
    result_SVM, stdev_SVM = moyenne(tableau_SVM)

    methode = (["0.1", "1", "10"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index', 'F1 Score']


    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_SVM[i][s][m] for i in range(len(methode))]
            standev = [stdev_SVM[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","SVM")

def effacer_fichiers_graph():
    dossier = "./Graph/"  
    
    if not os.path.exists(dossier):
        print(f"Le dossier {dossier} n'existe pas.")
        return
    
    for fichier in os.listdir(dossier):
        chemin_fichier = os.path.join(dossier, fichier)
        
        if os.path.isfile(chemin_fichier):
            os.remove(chemin_fichier)

def sauvergarder(df,filename):
    test = multiple(df,5,True)
    with open(filename, "wb") as fichier:
        pickle.dump(test, fichier)

def sauvergarder_hyper(df,filename):
    test = multiple(df,5,False)
    with open(filename, "wb") as fichier:
        pickle.dump(test, fichier)

def analyse_dataframe(df, yN, sensible, unprivilege):
    # Récupération du nom de la variable passée
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    df_name = [var_name for var_name, var_val in callers_local_vars if var_val is df]
    df_name = df_name[0] if df_name else "df"

    # Taille du dataframe
    nb_rows, nb_cols = df.shape
    print(f"Voici {df_name} avec {nb_rows} lignes et {nb_cols} colonnes.")

    # Pourcentage de 1 dans la colonne yN
    if yN in df.columns:
        pct_yN_1 = (df[yN] == 1).mean() * 100
        print(f"Pourcentage de 1 dans la colonne '{yN}': {pct_yN_1:.2f}%")
    else:
        print(f"Colonne '{yN}' non trouvée dans {df_name}.")

    # Analyse des colonnes sensibles
    print("\nAnalyse des colonnes sensibles (unprivileged values) :")
    for col, unpriv_vals in zip(sensible, unprivilege):
        if col in df.columns:
            pct_unpriv = df[col].isin(unpriv_vals).mean() * 100
            print(f" - {col} : {pct_unpriv:.2f}% des valeurs sont dans {unpriv_vals} (unprivileged)")
        else:
            print(f" - {col} : colonne non trouvée dans {df_name}")

def bank():
    df_bank = takeDataframe('bank-full.csv', 'utf-8')
    df_bankN = numeric(df_bank)
    global sensible, privilege, unprivilege, methode
    sensible = (['maritalN','educationN'])
    privilege = ([[2], [3]])
    unprivilege = ([[0, 1], [0, 1, 2]])
    print_disparate_impact(df_bankN, "yN", sensible, unprivilege, privilege)
    analyse_dataframe(df_bankN, "yN", sensible, unprivilege)
    sauvergarder(df_bankN,"Bank.pkl")
    sauvergarder_hyper(df_bankN,"Bank_hyper.pkl")

def student():
    df_student = takeDataframe('student-por.csv', 'utf-8')
    df_studentN = numerique(df_student)
    global sensible, privilege, unprivilege, methode
    sensible = (['sexN', 'ageN'])
    privilege = ([[1], [1]])
    unprivilege = ([[0], [0]])
    print_disparate_impact(df_studentN, "yN", sensible, unprivilege, privilege)
    analyse_dataframe(df_studentN, "yN", sensible, unprivilege)
    sauvergarder(df_studentN,"Student.pkl")
    sauvergarder_hyper(df_studentN,"Student_hyper.pkl")

def law():
    df_law = takeDataframe('lawschool.csv', 'utf-8')
    df_law = df_law.dropna(axis=0)
    df_law = df_law.rename(columns={'bar1': 'yN'})
    df_law.drop(columns=['race1', 'race2', 'race3', 'race4', 'race5', 'race6', 'race8'], inplace=True)
    global sensible, privilege, unprivilege, methode
    sensible = (['gender', 'race7'])
    privilege = ([[1], [1]])
    unprivilege = ([[0], [0]])
    print_disparate_impact(df_law, "yN", sensible, unprivilege, privilege)
    analyse_dataframe(df_law, "yN", sensible, unprivilege)
    sauvergarder(df_law,"Law.pkl")
    sauvergarder_hyper(df_law,"Law_hyper.pkl")

def german():
    df_german = takeDataframe('german_credit_risk.csv', 'utf-8')
    df_german = df_german.drop(columns=['Id'])
    df_german['Saving accounts'] = df_german['Saving accounts'].fillna('Unknown')
    df_german['Checking account'] = df_german['Checking account'].fillna('Unknown')
    df_germanN = numerique_german(df_german)
    global sensible, privilege, unprivilege, methode
    sensible = (['sexN', 'AgeN'])
    privilege = ([[1], [1]])
    unprivilege = ([[0], [0]])
    print_disparate_impact(df_germanN, "yN", sensible, unprivilege, privilege)
    analyse_dataframe(df_germanN, "yN", sensible, unprivilege)
    sauvergarder(df_germanN,"German.pkl")
    sauvergarder_hyper(df_germanN,"German_hyper.pkl")

def adult():
    df_adult = takeDataframe('adult.data.txt', 'utf-8')
    df_adult = df_adult.drop(columns=['fnlwgt'])
    df_adultN = numerique_adult(df_adult)
    global sensible, privilege, unprivilege, methode
    sensible = (['sexN','raceN'])
    privilege = ([[0], [0]])
    unprivilege = ([[1], [1,2,3,4]])
    print_disparate_impact(df_adultN, "yN", sensible, unprivilege, privilege)
    analyse_dataframe(df_adultN, "yN", sensible, unprivilege)
    sauvergarder(df_adultN,"Adult.pkl")
    sauvergarder_hyper(df_adultN,"Adult_hyper.pkl")

def default():
    df_default = takeDataframe('default_credit.csv', 'utf-8')
    df_default.drop(columns=['ID'], inplace=True)
    global sensible, privilege, unprivilege, methode
    sensible = (['SEX', 'MARRIAGE'])
    privilege = ([[1], [1]])       # Male, Married
    unprivilege = ([[2], [0, 2, 3]])
    print_disparate_impact(df_default, "yN", sensible, unprivilege, privilege)
    analyse_dataframe(df_default, "yN", sensible, unprivilege)
    sauvergarder(df_default, "Default.pkl")
    sauvergarder_hyper(df_default, "Default_hyper.pkl")

def plot_wilcoxon_heatmap(result_bank, result_student, result_law, result_german, result_adult, result_default):
    columns = ['Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index', 'F1 Score']
    all_data = [result_bank, result_student, result_law, result_german, result_adult, result_default]
    classifiers = ["Naive Bayes", "Logistic Regression", "KNN", "Random Forest", "Neural Network", "SVM"]
    
    functions = [
        lambda x: x.disparate_impact(),
        lambda x: x.equal_opportunity_difference(),
        lambda x: x.accuracy(),
        lambda x: x.error_rate_difference(),
        lambda x: (x.true_negative_rate() + x.true_positive_rate()) / 2,
        lambda x: x.generalized_entropy_index(),
        lambda x: 2 * x.precision() * x.recall() / (x.precision() + x.recall())
    ]

    for m in range(7):
        g = [[] for _ in range(6)]
        for data in all_data:
            for i in range(6):
                g[i].extend([functions[m](data[j][i][0]) for j in range(5)])  # sensible index 0

        matrix = np.full((6,6), np.nan)
        for i in range(6):
            for j in range(i+1, 6):
                stat, p = wilcoxon(g[i], g[j])
                matrix[i][j] = p
                matrix[j][i] = p

        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, xticklabels=classifiers, yticklabels=classifiers, annot=True, cmap='coolwarm', fmt=".3f")
        plt.title(f"P-values Wilcoxon - {columns[m]}")
        plt.tight_layout()
        plt.savefig(f"./Graph/wilcoxon_{columns[m].replace(' ', '_')}.png")
        plt.close()

def collect_all_metrics(metric_index):
    all_data = []
    model_names = ["NB", "LR", "KNN", "RF", "NN", "SVM"]

    all_results = [
        ("bank", result_bank),
        ("student", result_student),
        ("law", result_law),
        ("german", result_german),
        ("adult", result_adult),
        ("default", result_default)
    ]

    for name, dataset in all_results:
        for sensitive_index in range(2): 
            for fold in range(5): 
                for model in range(6): 
                    z = dataset[fold][model][sensitive_index]
                    precision = z.precision()
                    recall = z.recall()
                    f1 = 2 * precision * recall / (precision + recall) 

                    metric = [
                        z.disparate_impact(),
                        z.equal_opportunity_difference(),
                        z.accuracy(),
                        z.error_rate_difference(),
                        (z.true_negative_rate() + z.true_positive_rate()) / 2,
                        z.generalized_entropy_index(),
                        f1
                    ][metric_index]
                    all_data.append({
                        "Model": model_names[model],
                        "Score": metric,
                        "Fold": fold,
                        "Sensitive Index": sensitive_index,
                        "Dataset": name
                    })
    
    df = pd.DataFrame(all_data)
    
    print(df.head())
    
    df_wide = df.pivot(index=["Dataset", "Fold", "Sensitive Index"], columns="Model", values="Score")
    
    return df_wide

def generate_global_cd_diagram(metric_index, title):
    if metric_index in [1, 3]:
        df_metrics = collect_all_metrics(metric_index).applymap(abs)
    elif metric_index == 0 :
        df_metrics = collect_all_metrics(metric_index).applymap(lambda x: abs(math.log(x)))
    else :
        df_metrics = collect_all_metrics(metric_index)
    if metric_index in [0, 1, 3, 5] :
        result = autorank(df_metrics, alpha=0.05, verbose=True, order='ascending')
    else :
        result = autorank(df_metrics, alpha=0.05, verbose=True)
    plot_stats(result,allow_insignificant = True)
    plt.savefig(f"./Graph/nemenyi_{title.replace(' ', '_').replace(':', '').replace('-', '')}.png")
    plt.close()

def print_disparate_impact(df, target, sensible_attrs, unprivileged_vals, privileged_vals):
    print("== Disparate Impact Initial ==")
    for attr, unpriv, priv in zip(sensible_attrs, unprivileged_vals, privileged_vals):
        prot_group = df[df[attr].isin(unpriv)]
        priv_group = df[df[attr].isin(priv)]

        p_prot = (prot_group[target] == 1).mean()
        p_priv = (priv_group[target] == 1).mean()

        if p_priv == 0:
            di = float('inf')
        else:
            di = p_prot / p_priv

        print(f"{attr} → DI = {di:.3f} (prot.={p_prot:.3f}, priv.={p_priv:.3f})")
    print()

#bank()
student()
#law()
#german()
#adult()
#default() 

effacer_fichiers_graph()

with open("Bank.pkl", "rb") as fichier:
    result_bank = pickle.load(fichier)
    sensible = (['maritalN','educationN'])
calcul(result_bank, "Bank")

with open("Bank_hyper.pkl", "rb") as fichier:
    result_bank_hyper = pickle.load(fichier)
    sensible = (['maritalN','educationN'])
calcul_hyperparametre(result_bank_hyper, "Bank")

with open("Student.pkl", "rb") as fichier:
    result_student = pickle.load(fichier)
    sensible = (['sexN', 'ageN'])
calcul(result_student, "Student")

with open("Student_hyper.pkl", "rb") as fichier:
    result_student_hyper = pickle.load(fichier)
    sensible = (['sexN', 'ageN'])
calcul_hyperparametre(result_student_hyper, "Student")

with open("Law.pkl", "rb") as fichier:
    result_law = pickle.load(fichier)
    sensible = (['gender', 'race7'])
calcul(result_law, "Law")

with open("Law_hyper.pkl", "rb") as fichier:
    result_law_hyper = pickle.load(fichier)
    sensible = (['gender', 'race7'])
calcul_hyperparametre(result_law_hyper, "Law")

with open("German.pkl", "rb") as fichier:
    result_german = pickle.load(fichier)
    sensible = (['sexN', 'AgeN'])
calcul(result_german, "German")

with open("German_hyper.pkl", "rb") as fichier:
    result_german_hyper = pickle.load(fichier)
    sensible = (['sexN', 'AgeN'])
calcul_hyperparametre(result_german_hyper, "German")

with open("Adult.pkl", "rb") as fichier:
    result_adult = pickle.load(fichier)
    sensible = (['sexN','raceN'])
calcul(result_adult, "Adult")

with open("Adult_hyper.pkl", "rb") as fichier:
    result_adult_hyper = pickle.load(fichier)
    sensible = (['sexN','raceN'])
calcul_hyperparametre(result_adult_hyper, "Adult")

with open("Default.pkl", "rb") as fichier:
    result_default = pickle.load(fichier)
    sensible = (['SEX', 'MARRIAGE'])
calcul(result_default, "Default")

with open("Default_hyper.pkl", "rb") as fichier:
    result_default_hyper = pickle.load(fichier)
    sensible = (['SEX', 'MARRIAGE'])
calcul_hyperparametre(result_default_hyper, "Default")

plot_wilcoxon_heatmap(result_bank, result_student, result_law, result_german, result_adult, result_default)

generate_global_cd_diagram(metric_index=0, title="Nemenyi Diagram - Disparate Impact across all datasets & attributes")
generate_global_cd_diagram(metric_index=1, title="Nemenyi Diagram - Equal Opportunity Difference across all datasets & attributes")
generate_global_cd_diagram(metric_index=2, title="Nemenyi Diagram - Accuracy across all datasets & attributes")
generate_global_cd_diagram(metric_index=3, title="Nemenyi Diagram - Error Rate Difference across all datasets & attributes")
generate_global_cd_diagram(metric_index=4, title="Nemenyi Diagram - Balance Accuracy across all datasets & attributes")
generate_global_cd_diagram(metric_index=5, title="Nemenyi Diagram - Generalized entropy index across all datasets & attributes")
generate_global_cd_diagram(metric_index=6, title="Nemenyi Diagram - F1 Score across all datasets & attributes")