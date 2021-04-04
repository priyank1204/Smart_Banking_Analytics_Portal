import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sqlalchemy import column
from sklearn.preprocessing import OneHotEncoder
import joblib

#=======================================================================================================================
class Fraud_preprocess:
    '''
    Description: contains method for preprocessing of data. This class is being called in prediction.py file.
    Inputs for methods: methods takes data which is to be processed as input.
    Output for methods: return preprocesed data in form of dataframe.
    '''
    def initialize_columns(self, data):
        data.columns = ['step', 'customer', 'age', 'gender', 'zipcodeOri', 'merchant',
                        'zipMerchant', 'category', 'amount']
        return data

    def drop_columns(self, data):
        data_reduced = data.drop(['zipcodeOri', 'zipMerchant'], axis=1)
        return data_reduced

    def obj_to_cat(self, data_reduced):
        col_categorical = data_reduced.select_dtypes(include=['object']).columns
        for col in col_categorical:
            data_reduced[col] = data_reduced[col].astype('category')
        # categorical values ==> numeric values
        data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)
        return data_reduced
# ======================================================================================================================
class LA_preprocess:
    '''
    Description: contains method for preprocessing of data. This class is being called in prediction.py file.
    Inputs for methods: methods takes data which is to be processed as input.
    Output for methods: return preprocesed data in form of dataframe.
    '''
    def initialize_columns(self, data):  # depend ......................
        data.columns = ['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
                        'Education', 'Mortgage', 'Securities Account',
                           'CD Account', 'Online', 'CreditCard']
        return data

    def drop_columns(self, data):
        new_data = data.drop(["ID","ZIP Code"], axis=1)
        return new_data

    def encoder(self,data):
        le = LabelEncoder()
        cat_cols = ['Family', 'Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
        data[cat_cols] = data[cat_cols].apply(le.fit_transform)
        return data
#=======================================================================================================================
class LR_preprocess:
    '''
    Description: contains method for preprocessing of data. This class is being called in prediction.py file.
    Inputs for methods: methods takes data which is to be processed as input.
    Output for methods: return preprocesed data in form of dataframe.
    '''
    def initialize_columns(self, data):
        data.columns = ['RowID','Loan_Amount', 'Term', 'Interest_Rate', 'Employment_Years','Home_Ownership',
         'Annual_Income', 'Verification_Status','Loan_Purpose', 'State', 'Debt_to_Income', 'Delinquent_2yr',
         'Revolving_Cr_Util', 'Total_Accounts','Longest_Credit_Length']
        return data

    def drop_col(self,data):
        return data.drop(columns=['RowID'])

    def feature_engg(self,data):
        data['Term']= data['Term'].str.extract('(\d+)',expand=False)
        data['Term'] = pd.to_numeric(data['Term'])

        cols = ['Home_Ownership','Verification_Status','Loan_Purpose','State']
        data[cols]=data[cols].fillna(data.mode().iloc[0])

        le = LabelEncoder()
        data[cols]=data[cols].apply(le.fit_transform)
        return data

    def outlier_removal(self,data):
        def outlier_limits(col):
            Q3, Q1 = np.nanpercentile(col, [75,25])
            IQR= Q3-Q1
            UL= Q3+1.5*IQR
            LL= Q1-1.5*IQR
            return UL, LL

        for column in data.columns:
            if data[column].dtype != 'int64':
                UL, LL= outlier_limits(data[column])
                data[column]= np.where((data[column] > UL) | (data[column] < LL), np.nan, data[column])

        return data

    def imputer(self,data):
        df_mice = data.copy()

        mice= IterativeImputer(random_state=101)
        df_mice.iloc[:,:]= mice.fit_transform(df_mice)
        return df_mice
#=======================================================================================================================
class LE_preprocess:
    '''
        Description: contains method for preprocessing of data. This class is being called in prediction.py file.
        Inputs for methods: methods takes data which is to be processed as input.
        Output for methods: return preprocesed data in form of dataframe.
        '''
    def initialize_columns(self, data):
        data.columns = ['Current Loan Amount', 'Credit Score', 'Annual Income',
        'Years in current job', 'Monthly Debt', 'Years of Credit History',
        'Months since last delinquent', 'Number of Open Accounts',
        'Number of Credit Problems', 'Current Credit Balance',
        'Maximum Open Credit', 'Term_Long Term']
        return data
#=======================================================================================================================
class MA_preprocess:
    '''
        Description: contains method for preprocessing of data. This class is being called in prediction.py file.
        Inputs for methods: methods takes data which is to be processed as input.
        Output for methods: return preprocesed data in form of dataframe.
        '''
    def initialize_columns(self,data):
        data.columns = ["CONCAT","postcode","Qtr","unit"]
        return data

    def drop_columns(self,data):
        return data.drop(columns=['CONCAT'])
#=======================================================================================================================
class MS_preprocess:
    '''
        Description: contains method for preprocessing of data. This class is being called in prediction.py file.
        Inputs for methods: methods takes data which is to be processed as input.
        Output for methods: return preprocesed data in form of dataframe.
        '''
    def rename(self,data):
        data.rename(columns={'default': 'credit_default', 'housing': 'housing_loan', 'loan': 'personal_loan',
                             'day': 'last_contacted_day', 'month': 'last_contacted_month',
                             'duration': 'last_call_duration',
                             'campaign': 'contacts_during_camapign', 'pdays': 'days_passed',
                             'previous': 'contacts_before_campaign',
                             'y': 'deposit'}, inplace=True)
        return data

    def columns_match(self,data):
        columns = joblib.load('pickle_files/ms_cols.txt')
        # Get missing columns in the training test
        missing_cols = set(columns) - set( data.columns )
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            data[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        data = data[columns]
        return data

    def drop_columns(self, data):
        new_data = data.drop(
            ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'contact', 'poutcome'],axis=1)
        return new_data

    # creating categories out of days_passed column
    def days_passed(self,data):
        data['days_passed'] = np.where(data['days_passed']>=50, 'never_contacted', 'recent')
        return data

    # converting education col to categories
    def job(self, data):
        data['job'].replace(['services','admin.','blue-collar','technician','management','housemaid'],'salaried',inplace = True)
        data['job'].replace(['self-employed' ,'entrepreneur'],'self-employed',inplace = True)
        return data

    def education_cat(self, data):
        data['education'].replace(['basic.4y', 'basic.6y'],'primary',inplace = True)
        data['education'].replace(['high.school', 'basic.9y'],'secondary',inplace = True)
        data['education'].replace(['professional.course', 'university.degree'],'tertiary',inplace = True)
        return data

    def contacted_month(self, data):
        data['last_contacted_month'].replace(['jan', 'feb', 'mar', 'apr'],'jan-april',inplace = True)
        data['last_contacted_month'].replace(['may', 'jun', 'jul', 'aug'],'may-aug',inplace = True)
        data['last_contacted_month'].replace(['sep', 'oct', 'nov', 'dec'],'sep-dec',inplace = True)
        return data

    def contacts_before_campaign(self, data):
    # converting contacts_before_campaign to categories
        def toCategorical_contacts_before_campaign(x):
            if (x > 0 and x < 10):
                return '<10'
            else:
                return '0'
        data['contacts_before_campaign'] = data.contacts_before_campaign.apply(toCategorical_contacts_before_campaign)
        data['last_call_duration'] = data['last_call_duration'].apply(lambda n:n/60).round(2)
        return data

    def encoding(self,data):
        cat_cols = ['job', 'marital', 'education', 'credit_default', 'housing_loan',
                              'personal_loan', 'last_contacted_month', 'day_of_week', 'days_passed',
                              'contacts_before_campaign']
        encoder=OneHotEncoder()
        encoded_data=encoder.fit_transform(data[cat_cols])
        data1=pd.DataFrame(encoded_data.toarray())
        data=data.drop(cat_cols,axis=1)
        data=pd.concat([data,data1],axis=1)
        return data
#=======================================================================================================================
class LPR_preprocess:
    '''
        Description: contains method for preprocessing of data. This class is being called in prediction.py file.
        Inputs for methods: methods takes data which is to be processed as input.
        Output for methods: return preprocesed data in form of dataframe.
        '''
    def initialise_columns(self, data):
        return data

    def fill_cols(self, columns, test):
        # Get missing columns in the training test
        missing_cols = set(columns) - set(test.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            test[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        test = test[columns]
        return test

    def error_flag_column(self, data):
        data['DAYS_EMPLOYED_ERROR'] = data["DAYS_EMPLOYED"] == 365243
        # Replace the error values with nan
        data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
        return data

    def new_columns(self, data):
        data['INCOME_GT_CREDIT_FLAG'] = data['AMT_INCOME_TOTAL'] > data['AMT_CREDIT']
        # Column to represent Credit Income Percent
        data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
        # Column to represent Annuity Income percent
        data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        # Column to represent Credit Term
        data['CREDIT_TERM'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
        # Column to represent Days Employed percent in his life
        data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
        return data

    def joining_berau_application(self, bureau, application):
        # Combining numerical features
        grp = bureau.drop(['SK_ID_BUREAU'], axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
        grp.columns = ['BUREAU_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        application_bureau = application.merge(grp, on='SK_ID_CURR', how='left')
        application_bureau.update(application_bureau[grp.columns].fillna(0))
        # Combining categorical features
        bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))
        bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
        grp = bureau_categorical.groupby(by=['SK_ID_CURR']).mean().reset_index()
        grp.columns = ['BUREAU_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
        application_bureau.update(application_bureau[grp.columns].fillna(0))
        columns = joblib.load(r'pickle_files\bureau_columns.txt')
        application_bureau = self.fill_cols(columns, application_bureau)
        return application_bureau

    def feature_engineering(self, bureau, application_bureau):
        # Number of past loans per customer
        grp = bureau.groupby(by=['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(
            columns={'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})
        application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
        application_bureau['BUREAU_LOAN_COUNT'] = application_bureau['BUREAU_LOAN_COUNT'].fillna(0)
        # Number of types of past loans per customer
        grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])[
            'CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
        application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
        application_bureau['BUREAU_LOAN_TYPES'] = application_bureau['BUREAU_LOAN_TYPES'].fillna(0)
        # Debt over credit ratio
        bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
        bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
        grp1 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])[
            'AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})
        grp2 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
            'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CREDIT_SUM_DEBT'})
        grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT'] / grp1['TOTAL_CREDIT_SUM']
        del grp1['TOTAL_CREDIT_SUM']
        application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
        application_bureau['DEBT_CREDIT_RATIO'] = application_bureau['DEBT_CREDIT_RATIO'].fillna(0)
        application_bureau['DEBT_CREDIT_RATIO'] = application_bureau.replace([np.inf, -np.inf], 0)
        application_bureau['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau['DEBT_CREDIT_RATIO'],
                                                                downcast='float')
        # Overdue over debt ratio
        bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
        bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
        grp1 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])[
            'AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(
            columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
        grp2 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
            'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
        grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE'] / grp2['TOTAL_CUSTOMER_DEBT']
        del grp1['TOTAL_CUSTOMER_OVERDUE']
        application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
        application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau['OVERDUE_DEBT_RATIO'].fillna(0)
        application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau.replace([np.inf, -np.inf], 0)
        application_bureau['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau['OVERDUE_DEBT_RATIO']
                                                             ,downcast='float')
        return application_bureau

    def joining_previousapplication_to_applicationbereau(self, previous_applicaton, application_bureau):
        # Number of previous applications per customer
        grp = previous_applicaton[['SK_ID_CURR', 'SK_ID_PREV']].groupby(by=['SK_ID_CURR'])[
            'SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV': 'PREV_APP_COUNT'})
        application_bureau_prev = application_bureau.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev['PREV_APP_COUNT'] = application_bureau_prev['PREV_APP_COUNT'].fillna(0)
        # Combining numerical features
        grp = previous_applicaton.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
        prev_columns = ['PREV_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        grp.columns = prev_columns
        application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
        # Combining categorical features
        prev_categorical = pd.get_dummies(previous_applicaton.select_dtypes('object'))
        prev_categorical['SK_ID_CURR'] = previous_applicaton['SK_ID_CURR']
        prev_categorical.head()
        grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()
        grp.columns = ['PREV_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
        return application_bureau_prev
        ################################

    def Joining_POS_CASH_balance_to_application_bureau_prev_data(self, pos_cash, application_bureau_prev):
        # Combining numerical features
        grp = pos_cash.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
        prev_columns = ['POS_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        grp.columns = prev_columns
        application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
        # Combining categorical features
        pos_cash_categorical = pd.get_dummies(pos_cash.select_dtypes('object'))
        pos_cash_categorical['SK_ID_CURR'] = pos_cash['SK_ID_CURR']
        grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()
        grp.columns = ['POS_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
        columns = joblib.load(r'pickle_files\pos_cash_columns.txt')
        application_bureau_prev = self.fill_cols(columns, application_bureau_prev)
        return application_bureau_prev

    def joining_InstallmentsPaymentsdata_to_application_bureau_prev_data(self, insta_payments, application_bureau_prev):
        # Combining numerical features and there are no categorical features in this dataset
        grp = insta_payments.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
        prev_columns = ['INSTA_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        grp.columns = prev_columns
        application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
        return application_bureau_prev

    def Joining_Creditcardbalancedata_to_application_bureau_prev(self, application_bureau_prev, credit_card):
        # Combining numerical features
        grp = credit_card.drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()
        prev_columns = ['CREDIT_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        grp.columns = prev_columns
        application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
        # Combining categorical features
        credit_categorical = pd.get_dummies(credit_card.select_dtypes('object'))
        credit_categorical['SK_ID_CURR'] = credit_card['SK_ID_CURR']
        grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()
        grp.columns = ['CREDIT_' + column if column != 'SK_ID_CURR' else column for column in grp.columns]
        application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
        application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
        columns = joblib.load(r'pickle_files\crdit_card_columns.txt')
        application_bureau_prev = self.fill_cols(columns, application_bureau_prev)
        return application_bureau_prev

    def featurization(self, data):
        imputer_num = joblib.load(r'pickle_files\imputer_num.pkl')
        scaler_num = joblib.load(r'pickle_files\scaler_num.pkl')
        imputer_cat = joblib.load(r'pickle_files\imputer_cat.pkl')
        ohe = joblib.load(r'pickle_files\ohe.pkl')
        features = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        # Seperation of columns into numeric and categorical columns
        types = np.array([dt for dt in features.dtypes])
        all_columns = features.columns.values
        is_num = types != 'object'
        num_cols = all_columns[is_num]
        cat_cols = all_columns[~is_num]
        # Featurization of numeric data
        features_num = imputer_num.transform(features[num_cols])
        features_num1 = scaler_num.transform(features_num)
        features_num_final = pd.DataFrame(features_num1, columns=num_cols)
        # Featurization of categorical data
        features_cat = imputer_cat.transform(features[cat_cols])
        features_cat1 = pd.DataFrame(features_cat, columns=cat_cols)
        # one hot encoding
        features_cat2 = ohe.transform(features_cat1)
        cat_cols_ohe = list(ohe.get_feature_names(input_features=cat_cols))
        features_cat_final = pd.DataFrame(features_cat2, columns=cat_cols_ohe)
        # Final complete data
        features_final = pd.concat([features_num_final, features_cat_final], axis=1)
        return features_final

    # feature selection
    def feature_selection(self, data):
        data = data[['CREDIT_TERM', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2',
                     'DAYS_BIRTH', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED',
                     'POS_CNT_INSTALMENT_FUTURE', 'ANNUITY_INCOME_PERCENT',
                     'DAYS_REGISTRATION', 'DAYS_EMPLOYED_PERCENT', 'INSTA_AMT_PAYMENT',
                     'DAYS_LAST_PHONE_CHANGE', 'PREV_AMT_ANNUITY',
                     'BUREAU_DAYS_CREDIT_ENDDATE', 'CREDIT_INCOME_PERCENT',
                     'INSTA_NUM_INSTALMENT_NUMBER', 'BUREAU_DAYS_CREDIT',
                     'REGION_POPULATION_RELATIVE', 'BUREAU_AMT_CREDIT_SUM',
                     'DEBT_CREDIT_RATIO', 'PREV_SELLERPLACE_AREA',
                     'PREV_HOUR_APPR_PROCESS_START', 'PREV_DAYS_DECISION',
                     'BUREAU_DAYS_CREDIT_UPDATE', 'PREV_DAYS_LAST_DUE_1ST_VERSION',
                     'INSTA_DAYS_ENTRY_PAYMENT', 'PREV_AMT_CREDIT', 'LIVINGAREA_AVG']]
        return data