import pandas as pd
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

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
    # Recherche des statisques + moyenne
    taille = len(liste)
    resultat = [0]*len(liste[0])
    somme_carre = [0]*len(liste[0])
    n=0
    for x in liste[0] :
        resultat[n] = [0]*len(x)
        somme_carre[n] = [0]*len(x)
        for y in range(len(sensible)):
            resultat[n][y] = [0]*6
            somme_carre[n][y] = [0]*6
        n = n+1
    for x in liste:
        m = 0
        for y in x :
            o = 0
            for z in y :
                resultat[m][o][0] = resultat[m][o][0] + (z.disparate_impact()/taille)
                resultat[m][o][1] = resultat[m][o][1] + (z.equal_opportunity_difference()/taille)
                resultat[m][o][2] = resultat[m][o][2] + (z.accuracy()/taille)
                resultat[m][o][3] = resultat[m][o][3] + (z.error_rate_difference()/taille)
                resultat[m][o][4] = resultat[m][o][4] + (((z.true_negative_rate()/taille) + (z.true_positive_rate()/taille))/2)
                resultat[m][o][5] = resultat[m][o][5] + (z.generalized_entropy_index()/taille)


                somme_carre[m][o][0] += (z.disparate_impact() ** 2) / taille
                somme_carre[m][o][1] += (z.equal_opportunity_difference() ** 2) / taille
                somme_carre[m][o][2] += (z.accuracy() ** 2) / taille
                somme_carre[m][o][3] += (z.error_rate_difference() ** 2) / taille
                somme_carre[m][o][4] += ((((z.true_negative_rate()/taille) + (z.true_positive_rate()/taille))/2) ** 2) / taille
                somme_carre[m][o][5] += (z.generalized_entropy_index() ** 2) / taille
                o = o+1
            m = m+1

    ecart_type = [0]*len(liste[0])
    for n in range(len(resultat)):
        ecart_type[n] = [0]*len(resultat[n])
        for y in range(len(resultat[n])):
            ecart_type[n][y] = [0]*6
            for stat in range(4):
                variance = somme_carre[n][y][stat] - (resultat[n][y][stat] ** 2)
                ecart_type[n][y][stat] = math.sqrt(variance)

    return resultat, ecart_type

def plot_bar(test_name, x, y, stdev, bidule, name):
    plt.figure()
    plt.bar(x, y, yerr=stdev, width=0.4)
    plt.xlabel("Methode")
    plt.ylabel(test_name)
    plt.title("Methode " + test_name)
    plt.savefig(f"../Graph/{name}_{bidule}_{test_name}.png")
    plt.close()

def plot_bar_parametre (test_name, x, y, stdev, bidule, name, methode):
    plt.figure()
    plt.bar(x, y, yerr=stdev, width=0.4)
    plt.xlabel(methode)
    plt.ylabel(test_name)
    plt.title(methode + test_name)
    plt.savefig(f"../Graph/{name}_{bidule}_{test_name}_{methode}.png")
    plt.close()

def calcul(test,name):
    result, stdev = moyenne(test)

    # Tableau avec les resultats
    methode = (["naive_bayes", "logistic_regression", "k_neighbors", "random_forest", "neural_network", "support_vector_machine"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index']

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
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index']

    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_NB[i][s][m] for i in range(len(methode))]
            standev = [stdev_NB[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","Naive_Bayes")

            tableau_NB = [sous_tableau[0] for sous_tableau in test]
    
    tableau_KNN = [sous_tableau[1] for sous_tableau in test]        
    result_KNN, stdev_KNN = moyenne(tableau_KNN)

    methode = (["5", "10", "20"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index']

    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_KNN[i][s][m] for i in range(len(methode))]
            standev = [stdev_KNN[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","k_neighbors")

    tableau_RF = [sous_tableau[2] for sous_tableau in test]        
    result_RF, stdev_RF = moyenne(tableau_RF)

    methode = (["50", "100", "300"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index']

    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_RF[i][s][m] for i in range(len(methode))]
            standev = [stdev_RF[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","Random_Forests")

    tableau_NN = [sous_tableau[3] for sous_tableau in test]        
    result_NN, stdev_NN = moyenne(tableau_NN)

    methode = (["(100,)","(50, 50)" ,"(100, 50, 30)"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index']

    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_NN[i][s][m] for i in range(len(methode))]
            standev = [stdev_NN[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","Neural_networks")

    tableau_SVM = [sous_tableau[4] for sous_tableau in test]        
    result_SVM, stdev_SVM = moyenne(tableau_SVM)

    methode = (["0.1", "1", "10"])
    columns = ["Methode_sensible",'Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index']

    for s in range(len(sensible)):
        for m in range(len(columns)-1):
            y = [result_SVM[i][s][m] for i in range(len(methode))]
            standev = [stdev_SVM[i][s][m] for i in range(len(methode))]
            plot_bar_parametre(columns[m+1],methode,y,standev,sensible[s],name+"_parametre","SVM")

def effacer_fichiers_graph():
    dossier = "../Graph/"  
    
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

def bank():
    df_bank = takeDataframe('bank-full.csv', 'utf-8')
    df_bankN = numeric(df_bank)
    global sensible, privilege, unprivilege, methode
    sensible = (['maritalN','educationN'])
    privilege = ([[2], [3]])
    unprivilege = ([[0, 1], [0, 1, 2]])
    sauvergarder(df_bankN,"Bank.pkl")
    sauvergarder_hyper(df_bankN,"Bank_hyper.pkl")

def student():
    df_student = takeDataframe('student-por.csv', 'utf-8')
    df_studentN = numerique(df_student)
    global sensible, privilege, unprivilege, methode
    sensible = (['sexN', 'ageN'])
    privilege = ([[1], [1]])
    unprivilege = ([[0], [0]])
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
    sauvergarder(df_adultN,"Adult.pkl")
    sauvergarder_hyper(df_adultN,"Adult_hyper.pkl")

def wilson_coxon(data1, data2, data3, data4, data5):
    columns = ['Disparate Impact','Equal Opportunity Difference', 'Accuracy', 'Error Rate Difference', 'Balance Accuracy', 'Generalized entropy index']
    all_data = [data1, data2, data3, data4, data5]
    classifiers = ["naive_bayes", "logistic_regression", "k_neighbors", "random_forest", "neural_network", "support_vector_machine"]
    
    functions = [
        lambda x: x.disparate_impact(),
        lambda x: x.equal_opportunity_difference(),
        lambda x: x.accuracy(),
        lambda x: x.error_rate_difference(),
        lambda x: (x.true_negative_rate() + x.true_positive_rate())/2,
        lambda x: x.generalized_entropy_index()
    ]
    
    for m in range(6):
        g1 = []
        g2 = []
        g3 = []
        g4 = []
        g5 = []
        g6 = []
        g = [g1, g2, g3, g4, g5, g6]
        for i, data in enumerate(all_data, start=1):
            for n in range(2):
                for j in range(5):
                    g1.append(functions[m](data[j][0][n]))
                    g2.append(functions[m](data[j][1][n]))
                    g3.append(functions[m](data[j][2][n]))
                    g4.append(functions[m](data[j][3][n]))
                    g5.append(functions[m](data[j][4][n]))
                    g6.append(functions[m](data[j][5][n]))
        tab = np.empty((6, 6), dtype=object)  
        for i in range(6):
            for j in range(6):
                if i < j: 
                    stat, p_value = wilcoxon(g[i], g[j])
                    tab[i, j] = p_value
                else:
                    tab[i, j] = None  
        
        print(f"Metric: {columns[m]}")
        print("\t\t" + "\t".join(classifiers))
        for i in range(6):
            row = [classifiers[i]] + [f"{tab[i, j]:.4f}" if tab[i, j] is not None else "N/A" for j in range(6)]
            print("\t".join(row)) 
        
        print()

#bank()
student()
#law()
#german()
#adult()

effacer_fichiers_graph()

with open("Bank.pkl", "rb") as fichier:
    result_bank = pickle.load(fichier)
#calcul(result_bank,"bank")

with open("Student.pkl", "rb") as fichier:
    result_student = pickle.load(fichier)
calcul(result_student,"student")

with open("Law.pkl", "rb") as fichier:
    result_law = pickle.load(fichier)
#calcul(result_law,"law")

with open("German.pkl", "rb") as fichier:
    result_german = pickle.load(fichier)
#calcul(result_german,"german")
 
with open("Adult.pkl", "rb") as fichier:
    result_adult = pickle.load(fichier)
#calcul(result_adult,"adult")

wilson_coxon(result_bank, result_student, result_law, result_german, result_adult)
