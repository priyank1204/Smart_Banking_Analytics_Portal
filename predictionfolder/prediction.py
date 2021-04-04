from preprocessingfolder import preprocessingfile
import pandas as pd
import numpy as np
import joblib
import pickle
from Data_ingestion import data_ingestion
from data_visualization import data_visualization

class Fraud_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        '''
                  Description: This method takes the csv file from bulk_predict routes in app.py
                                           and calls pre defined preprocessing classes from prediction folder to give output file.
                  Output: Each method returns an output dataframe along with 4 chart names which are created and
                                           stored in static folder of main directory.
                  On Failure: Raise Exception.
         '''
        try:
            instance1 = data_ingestion.data_getter()
            data = instance1.data_load(file)

            instance2 = preprocessingfile.Fraud_preprocess()
            visuals = data_visualization.Data_Visualisation()
            visuals.delete_old_graphs('FD')

            set0 = instance2.initialize_columns(data)
            data_num = set0.select_dtypes(include='number')  # to get list of all numerical features in data
            num_col_ls = list(data_num.columns)

            set1 = instance2.drop_columns(set0)
            final_data = set1[['customer']]
            set2 = set1.drop(['customer'],axis=1)
            new_data = instance2.obj_to_cat(set2)
            # ============  pandas-prof. report ============================
            new_data.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
            # ==============================================================
            model_Fraud = instance1.decompress_pickle('pickle_files/Fraud_rf_new_model.pbz2')
            result = model_Fraud.predict(new_data)

            new_data['output'] = result
            chart3_name,imp_feature = visuals.feature_importance(new_data["output"],new_data.drop('output',axis=1),'FD')  # feature importance
            if imp_feature in num_col_ls:
                chart4_name=visuals.numeric_summary(new_data,imp_feature,'FD') # most important feature graph
            else:
                chart4_name = visuals.categorical_summary(new_data,imp_feature,'FD')

            new_data['output'] = np.where(new_data['output'] == 0, "Trusted", "Fraud")
            final_data['Output']=new_data['output']

            chart1_name = visuals.count_plot("Output", final_data, 'FD')  # count plot
            chart2_name = visuals.heat_map("output", new_data, 'FD')  # heat map
            return final_data, chart1_name, chart2_name,chart3_name,chart4_name
        except Exception as e:
            raise e
# ===================================================================================================
class LA_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        '''
                  Description: This method takes the csv file from LA_bulk_predict routes in app.py
                                           and calls pre defined preprocessing classes from prediction folder to give output file.
                  Output: Each method returns an output dataframe along with 4 chart names which are created and
                                           stored in static folder of main directory.
                  On Failure: Raise Exception.
         '''
        try:
            instance1 = data_ingestion.data_getter()
            data = instance1.data_load(file)
            instance2 = preprocessingfile.LA_preprocess()
            visuals = data_visualization.Data_Visualisation()
            visuals.delete_old_graphs('LA')

            set0 = instance2.initialize_columns(data)
            data_num = set0.select_dtypes(include='number') # to get list of all numerical features in data
            num_col_ls = list(data_num.columns)
            final_data = set0[['ID']]
            set1=instance2.drop_columns(set0)
            new_data = instance2.encoder(set1)
            # ============  pandas-prof. report ============================
            new_data.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
            # ==============================================================
            ss_LA = instance1.decompress_pickle('pickle_files/LA_Std_scaler.pbz2')
            model_LA = instance1.decompress_pickle('pickle_files/DTModel-1.pbz2')

            ss_result = ss_LA.transform(new_data)
            result = model_LA.predict(ss_result)

            new_data['output'] = result
            chart3_name,imp_feature = visuals.feature_importance(new_data["output"],new_data.drop('output',axis=1),'LA')  # feature importance
            if imp_feature in num_col_ls:
                chart4_name=visuals.numeric_summary(new_data,imp_feature,'LA') # most important feature graph
            else:
                chart4_name = visuals.categorical_summary(new_data,imp_feature,'LA')

            new_data['output'] = np.where(new_data['output'] == 0,"Rejected","Accepted")
            final_data['Output']=new_data['output']

            chart1_name=visuals.count_plot("Output",final_data,'LA')  # count plot
            chart2_name=visuals.heat_map("output",new_data,'LA')    # heat map
            return final_data,chart1_name,chart2_name,chart3_name,chart4_name
        except Exception as e:
            raise e

# =====================================================================================================
class LR_predict:
    def __init__(self):
        pass

    def predictor(self, file):
        '''
              Description: This method takes the csv file from LR_bulk_predict routes in app.py
                                       and calls pre defined preprocessing classes from prediction folder to give output file.
              Output: Each method returns an output dataframe along with 4 chart names which are created and
                                       stored in static folder of main directory.
              On Failure: Raise Exception.
     '''
        try:
            instance1 = data_ingestion.data_getter()
            data = instance1.data_load(file)
            instance2 = preprocessingfile.LR_preprocess()
            visuals = data_visualization.Data_Visualisation()
            visuals.delete_old_graphs('LR')

            set0 = instance2.initialize_columns(data)
            data_num = set0.select_dtypes(include='number')  # to get list of all numerical features in data
            num_col_ls = list(data_num.columns)

            set1 = instance2.drop_col(set0)
            set2 = instance2.feature_engg(set1)
            set3 = instance2.outlier_removal(set2)
            set4 = instance2.imputer(set3)
            # ============  pandas-prof. report ============================
            set4.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
            # ==============================================================
            lr_model = instance1.decompress_pickle('pickle_files/loan_risk.pbz2')

            result = lr_model.predict(set4)
            set4['output'] = result
            chart3_name,imp_feature = visuals.feature_importance(set4["output"],set4.drop('output',axis=1),'LR')  # feature importance
            if imp_feature in num_col_ls:
                chart4_name = visuals.numeric_summary(set4, imp_feature, 'LR')  # most important feature graph
            else:
                chart4_name = visuals.categorical_summary(set4,imp_feature,'LR')

            set4['output'] = np.where(set4['output'] == 0,"Risky","Safe")
            final_data = {'RowID':[i for i in set0['RowID']],'Output':[i for i in set4['output']]}

            chart1_name = visuals.count_plot("Output", final_data, 'LR')  # count plot
            chart2_name = visuals.heat_map("output", set4, 'LR')  # heat map
            print(chart3_name)
            print(chart4_name)
            return pd.DataFrame(final_data),chart1_name, chart2_name,chart3_name,chart4_name
        except Exception as e:
            raise e
# ===============================================================================================
class LE_predict:
    def __init__(self):
        pass

    def predictor(self, file):
        '''
                  Description: This method takes the csv file from LE_bulk_predict routes in app.py
                                           and calls pre defined preprocessing classes from prediction folder to give output file.
                  Output: Each method returns an output dataframe along with 4 chart names which are created and
                                           stored in static folder of main directory.
                  On Failure: Raise Exception.
         '''
        try:
            instance1 = data_ingestion.data_getter()
            data = instance1.data_load(file)
            instance2 = preprocessingfile.LE_preprocess()
            visuals = data_visualization.Data_Visualisation()
            visuals.delete_old_graphs('LE')

            data_final = instance2.initialize_columns(data)
            data_num = data_final.select_dtypes(include='number')  # to get list of all numerical features in data
            num_col_ls = list(data_num.columns)
            # ============  pandas-prof. report ============================
            data_final.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
            # ==============================================================
            le_model = instance1.decompress_pickle('pickle_files/LE-DecTreeModel.pbz2')

            result = le_model.predict(data_final)
            data_final['output'] = result
            chart3_name, imp_feature = visuals.feature_importance(data_final["output"], data_final.drop('output', axis=1),'LE')  # feature importance
            if imp_feature in num_col_ls:
                chart4_name = visuals.numeric_summary(data_final, imp_feature, 'LR')  # most important feature graph
            else:
                chart4_name = visuals.categorical_summary(data_final, imp_feature, 'LR')

            data_final['output'] = np.where(data_final['output'] == 0, "Not Eligible", "Eligible")

            data_final['RowID'] = pd.Series([i for i in range(len(data_final['output']))])

            final_data = {'RowID': [i for i in data_final['RowID']], 'Output': [i for i in data_final['output']]}
            chart1_name = visuals.count_plot("Output", final_data, 'LE')  # count plot
            chart2_name = visuals.heat_map("output", data_final, 'LE')  # heat map
            return pd.DataFrame(final_data), chart1_name, chart2_name ,chart3_name ,chart4_name
        except Exception as e:
            raise e
#=======================================================================================================================
class MA_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        '''
              Description: This method takes the csv file from MA_bulk_predict routes in app.py
                                       and calls pre defined preprocessing classes from prediction folder to give output file.
              Output: Each method returns an output dataframe along with 4 chart names which are created and
                                       stored in static folder of main directory.
              On Failure: Raise Exception.
     '''
        try:
            instance1 = data_ingestion.data_getter()
            data = instance1.data_load(file)
            instance2 = preprocessingfile.MA_preprocess()
            visuals = data_visualization.Data_Visualisation()
            visuals.delete_old_graphs('MA')

            new_data = instance2.initialize_columns(data)
            data_num = new_data.select_dtypes(include='number')  # to get list of all numerical features in data
            num_col_ls = list(data_num.columns)
            # ============  pandas-prof. report ============================
            new_data.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
            # ==============================================================
            final_data = new_data[['CONCAT']]
            new_data=instance2.drop_columns(new_data)
            model_MA= instance1.decompress_pickle('pickle_files/Mortgage_RE.pbz2')

            result = model_MA.predict(new_data)
            final_data['output'] = result
            return final_data
        except Exception as e:
            raise e
# =======================================================================================================================
class MS_predict():
    def __init__(self):
                pass

    def predictor(self, file):
        '''
                  Description: This method takes the csv file from MA_bulk_predict routes in app.py
                                           and calls pre defined preprocessing classes from prediction folder to give output file.
                  Output: Each method returns an output dataframe along with 4 chart names which are created and
                                           stored in static folder of main directory.
                  On Failure: Raise Exception.
         '''
        try:
            instance1 = data_ingestion.data_getter()
            data = instance1.data_load(file)

            instance2 = preprocessingfile.MS_preprocess()
            visuals = data_visualization.Data_Visualisation()
            visuals.delete_old_graphs('MS')

            set0 = instance2.rename(data)
            data_num = set0.select_dtypes(include='number')  # to get list of all numerical features in data
            num_col_ls = list(data_num.columns)

            set1 = instance2.drop_columns(set0)
            set2 = instance2.days_passed(set1)
            set3 = instance2.job(set2)
            set4 = instance2.education_cat(set3)
            set5 = instance2.contacted_month(set4)
            data_final = instance2.contacts_before_campaign(set5)

            data_final = pd.get_dummies(data_final,drop_first=True)
            data_final = instance2.columns_match(data_final)
            data_final = data_final[['last_call_duration','age','days_passed_recent','contacts_during_campaign','housing_loan_yes','personal_loan_yes','day_of_week_tue','last_contacted_month_may-aug','day_of_week_thu','marital_married','day_of_week_wed','day_of_week_mon','marital_single','contacts_during_campaign']]
            MS__model = instance1.decompress_pickle('pickle_files/MS_randomforest_model4.pbz2')
            result = MS__model.predict(data_final)
            data_final["output"]=result
            chart3_name, imp_feature = visuals.feature_importance(data_final["output"], data_final.drop('output', axis=1),'MS')  # feature importance
            if imp_feature in num_col_ls:
                chart4_name = visuals.numeric_summary(data_final, imp_feature, 'MS')  # most important feature graph
            else:
                chart4_name = visuals.categorical_summary(data_final, imp_feature, 'MS')

            final_result=pd.DataFrame(result, columns=['Output'])
            final_result["SrNo."]=np.arange(len(final_result["Output"]))
            pop_col=final_result.pop('Output')
            final_result["output"]=pop_col
            final_result['output'] = np.where(final_result['output'] == 0, "Not Subscribed", "Term Deposit")

            chart1_name = visuals.count_plot("output", final_result, 'MS')  # count plot
            chart2_name = visuals.heat_map("output", data_final, 'MS')  # heat map
            return final_result, chart1_name, chart2_name, chart3_name, chart4_name
        except Exception as e:
            raise e
#=======================================================================================================================
class LPR_predict():
    def _init_(self):
        pass

    def predictor(self,file):
        '''
                  Description: This method takes the csv file from PLR_bulk_predict routes in app.py
                                           and calls pre defined preprocessing classes from prediction folder to give output file.
                  Output: Each method returns an output dataframe along with 4 chart names which are created and
                                           stored in static folder of main directory.
                  On Failure: Raise Exception.
         '''
        try:
            instance1 = data_ingestion.data_getter()
            visuals = data_visualization.Data_Visualisation()
            visuals.delete_old_graphs('LPR')

            data = instance1.data_load(file[0])
            data1 = data[['SK_ID_CURR']]
            bureau=instance1.data_load(file[1])
            previos_application=instance1.data_load(file[5])
            pos_cash=instance1.data_load(file[4])
            insta_payments=instance1.data_load(file[3])
            credit_card=instance1.data_load(file[2])

            instance2 = preprocessingfile.LPR_preprocess()
            data=instance2.error_flag_column(data)
            data=instance2.new_columns(data)
            application_bureau=instance2.joining_berau_application(bureau,data)
            application_bureau=instance2.feature_engineering(bureau,application_bureau)
            application_bureau_prev=instance2.joining_previousapplication_to_applicationbereau(previos_application,application_bureau)
            application_bureau_prev=instance2.Joining_POS_CASH_balance_to_application_bureau_prev_data(pos_cash,application_bureau_prev)
            application_bureau_prev=instance2.joining_InstallmentsPaymentsdata_to_application_bureau_prev_data(insta_payments,application_bureau_prev)
            application_bureau_prev=instance2.Joining_Creditcardbalancedata_to_application_bureau_prev(application_bureau_prev,credit_card)
            application_bureau_prev=instance2.featurization(application_bureau_prev)
            application_bureau_prev=instance2.feature_selection(application_bureau_prev)

            data_num = application_bureau_prev.select_dtypes(include='number')  # to get list of all numerical features in data
            num_col_ls = list(data_num.columns)

            model = instance1.decompress_pickle(r'pickle_files\DecTreePLR.pbz2')
            output = model.predict(application_bureau_prev)
            application_bureau_prev['result'] = output
            chart3_name, imp_feature = visuals.feature_importance(application_bureau_prev["result"],
                                                                  application_bureau_prev.drop('result', axis=1),'LPR')  # feature importance
            if imp_feature in num_col_ls:
                chart4_name = visuals.numeric_summary(application_bureau_prev, imp_feature, 'LPR')  # most important feature graph
            else:
                chart4_name = visuals.categorical_summary(application_bureau_prev, imp_feature, 'LPR') # most imp feature categorical graph
            data1['result']=application_bureau_prev["result"]
            data1['result'] = np.where(data1['result'] == 0, "Loan Repayed", "Not Repayed")
            chart1_name = visuals.count_plot("result", data1,'LPR')  # count plot
            chart2_name = visuals.heat_map("result", application_bureau_prev, 'LPR')  # heat map
            return data1,chart1_name,chart2_name,chart3_name,chart4_name
        except Exception as e:
            raise e
