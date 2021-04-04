# import packages ...
from flask import Flask, render_template,url_for, request,redirect ,make_response
import pandas as pd
import warnings
import joblib
from flask_cors import cross_origin
from predictionfolder.prediction import LA_predict,Fraud_predict,LR_predict,LE_predict,MA_predict,MS_predict,LPR_predict
from pandas_profiling import ProfileReport
from Data_ingestion import data_ingestion
import matplotlib
matplotlib.use('Agg')

# function to remove warning messages ..
def warns(*args, **kwargs):
    pass
warnings.warn = warns

ALLOWED_EXTENSIONS = set(['csv','xlsx','data'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# instance for data ingestion class (to load compressed pickle file)..
pickle_load=data_ingestion.data_getter()

# load models stored in compressed pickle files ..
ss_LA = pickle_load.decompress_pickle('pickle_files/LA_Std_scaler.pbz2')
model_LA = pickle_load.decompress_pickle('pickle_files/DTModel-1.pbz2')
model_Fraud = pickle_load.decompress_pickle('pickle_files/Fraud_rf_new_model.pbz2')
model_LR = pickle_load.decompress_pickle('pickle_files/loan_risk.pbz2')
model_LE = pickle_load.decompress_pickle('pickle_files/LE-DecTreeModel.pbz2')
model_MA = pickle_load.decompress_pickle('pickle_files/Mortgage_RE.pbz2')
model_MS = pickle_load.decompress_pickle('pickle_files/MS_randomforest_model4.pbz2')

# instances of prediction.py file for bulk upload of different models ..
LA_instance = LA_predict()
Fraud_instance = Fraud_predict()
LR_instance = LR_predict()
LE_instance = LE_predict()
MA_instance = MA_predict()
MS_instance = MS_predict()
LPR_instance = LPR_predict()

# initializing flask app object ..
app = Flask(__name__)

@app.route('/')  # main route ..
@cross_origin()
def intro():
    return render_template('intro.html')

@app.route('/home',methods=['GET','POST']) # button in intro.html
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/LA',methods=['GET','POST']) # button in home.html
@cross_origin()
def LA():
    if request.method == 'POST':
        message='LA'
        return render_template('main_dashboard.html',message=message)

@app.route('/FD',methods=['GET','POST']) # button in home.html
@cross_origin()
def FD():
    if request.method == 'POST':
        message='FD'
        return render_template('main_dashboard.html',message=message)

@app.route('/LR',methods=['GET','POST']) # button in home.html
@cross_origin()
def LR():
    if request.method == 'POST':
        message='Lrisk'
        return render_template('main_dashboard.html',message=message)

@app.route('/LE',methods=['GET','POST']) # button in home.html
@cross_origin()
def LE():
    if request.method == 'POST':
        message='LE'
        return render_template('main_dashboard.html',message=message)

@app.route('/MA',methods=['GET','POST']) # button in home.html
@cross_origin()
def MA():
    if request.method == 'POST':
        message='MA'
        return render_template('main_dashboard.html',message=message)

@app.route('/MS',methods=['GET','POST']) # button in home.html
@cross_origin()
def MS():
    if request.method == 'POST':
        message='MS'
        return render_template('main_dashboard.html',message=message)

@app.route('/PLR',methods=['GET','POST']) # button in home.html
@cross_origin()
def PLR():
    if request.method == 'POST':
        message = 'PLR'
        return render_template('main_dashboard.html',message = message)

# route's button in home.html
@app.route('/LA_single_predict',methods=['GET','POST'])
@cross_origin()
def LA_single_predict():
    if request.method == 'POST':
        message='LA'
        return render_template('single_predict.html',message=message)
@app.route('/LA_multi_predict',methods=['GET','POST'])
@cross_origin()
def LA_multi_predict():
    if request.method == 'POST':
        message = 'LA'
        return render_template('multi_predict.html',message=message)
@app.route('/FD_single_predict',methods=['GET','POST'])
@cross_origin()
def FD_single_predict():
    if request.method == 'POST':
        message='FD'
        return render_template('single_predict.html',message=message)
@app.route('/FD_multi_predict',methods=['GET','POST'])
@cross_origin()
def FD_multi_predict():
    if request.method == 'POST':
        message = 'FD'
        return render_template('multi_predict.html',message=message)
@app.route('/lrisk_single_predict',methods=['GET','POST'])
@cross_origin()
def lrisk_single_predict():
    if request.method == 'POST':
        message='Lrisk'
        return render_template('single_predict.html',message=message)
@app.route('/lrisk_multi_predict',methods=['GET','POST'])
@cross_origin()
def lrisk_multi_predict():
    if request.method == 'POST':
        message = 'Lrisk'
        return render_template('multi_predict.html',message=message)
@app.route('/LE_single_predict',methods=['GET','POST'])
@cross_origin()
def LE_single_predict():
    if request.method == 'POST':
        message='LE'
        return render_template('single_predict.html',message=message)
@app.route('/LE_multi_predict',methods=['GET','POST'])
@cross_origin()
def LE_multi_predict():
    if request.method == 'POST':
        message = 'LE'
        return render_template('multi_predict.html',message=message)
@app.route('/MA_single_predict',methods=['GET','POST'])
@cross_origin()
def MA_single_predict():
    if request.method == 'POST':
        message='MA'
        return render_template('single_predict.html',message=message)
@app.route('/MA_multi_predict',methods=['GET','POST'])
@cross_origin()
def MA_multi_predict():
    if request.method == 'POST':
        message = 'MA'
        return render_template('multi_predict.html',message=message)
@app.route('/MS_single_predict',methods=['GET','POST'])
@cross_origin()
def Marketing_single_predict():
    if request.method == 'POST':
        message='MS'
        return render_template('single_predict.html',message=message)
@app.route('/MS_multi_predict',methods=['GET','POST'])
@cross_origin()
def Marketing_multi_predict():
    if request.method == 'POST':
        message = 'MS'
        return render_template('multi_predict.html',message=message)
@app.route('/PLR_multi_predict',methods=['GET','POST'])
@cross_origin()
def PLR_multi_predict():
    if request.method == 'POST':
        message = 'PLR'
        return render_template('multi_predict.html',message = message)

# Bulk Prediction Button's routes ... ( defined in multi_predict.html)
@app.route('/bulk_predict',methods=['GET','POST'])
@cross_origin()
def bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data,chart1,chart2,chart3,chart4= Fraud_instance.predictor(file)
            new_data = data[:8]
            message='FD'
            return render_template('bulk_output_file.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values,
                                   message=message,chart1_name=chart1,chart2_name=chart2,chart3_name=chart3,chart4_name=chart4)
        else:
            return redirect(request.url)
    else:
        return data.to_string() # to download output csv file in bulk predict..

@app.route('/LA_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LA_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data,chart1,chart2,chart3,chart4 = LA_instance.predictor(file)
            new_data = data[:8]  # to restrict number of rows in bulk output ..
            message = 'LA'
            return render_template('bulk_output_file.html', tables=[new_data.to_html(classes='data')],titles=data.columns.values,message=message,
                                   chart1_name=chart1,chart2_name=chart2,chart3_name=chart3,chart4_name=chart4)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/LR_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LR_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data,chart1,chart2,chart3,chart4 = LR_instance.predictor(file)
            new_data = data[:8]
            message = 'LR'
            return render_template('bulk_output_file.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values,
                                   message=message,chart1_name=chart1,chart2_name=chart2,chart3_name=chart3,chart4_name=chart4)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/LE_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LE_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data,chart1,chart2,chart3,chart4 = LE_instance.predictor(file)
            new_data = data[:8]
            message = 'LE'
            return render_template('bulk_output_file.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values,
                                   message=message,chart1_name=chart1,chart2_name=chart2,chart3_name=chart3,chart4_name=chart4)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/MA_bulk_predict',methods=['GET','POST'])
@cross_origin()
def MA_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data = MA_instance.predictor(file)
            new_data = data[:5]
            #message = 'MA'
            return render_template('result_bulk.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/MS_bulk_predict', methods=['GET', 'POST'])
@cross_origin()
def MS_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data,chart1,chart2,chart3,chart4 = MS_instance.predictor(file)
            new_data = data[:8]
            message='MS'
            return render_template('bulk_output_file.html', tables=[new_data.to_html(classes='data')],titles=data.columns.values,
                                message=message,chart1_name=chart1,chart2_name=chart2,chart3_name=chart3,chart4_name=chart4)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/PLR_bulk_predict',methods=['GET','POST'])
@cross_origin()
def PLR_bulk_predict():
    if request.method == "POST":
        file = request.files.getlist('file[]')
        if file:
            global data
            data,chart1,chart2,chart3,chart4 = LPR_instance.predictor(file)
            new_data=data[:8]
            new_data=pd.DataFrame(new_data)
            message = 'LPR'
        return render_template('bulk_output_file.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values,
                               message=message,chart1_name=chart1,chart2_name=chart2,chart3_name=chart3,chart4_name=chart4)
    else:
        return data.to_string()

# Single Prediction Button's routes ... ( defined in single_predict.html)
@app.route('/predict_FD',methods=['GET','POST'])
@cross_origin()
def predict_FD():
    if request.method == 'POST':
        Type = request.form.get("gender", False)
        if (Type == 'Male'):
            Type = 0
        elif (Type == 'Female'):
            Type = 1
        elif (Type == 'Enterprise'):
            Type = 2
        elif (Type == 'Unknown'):
            Type = 3

        amount = float(request.form.get("amount", False))
        merchant = float(request.form.get("merchant", False))
        category = float(request.form.get("category", False))
        step = float(request.form.get("step", False))
        age = float(request.form.get("age", False))

        df = pd.DataFrame(
            {"step": step ,"age": age,'gender': Type,  "merchant": merchant,
                    "category": category,"amount": amount  }, index=[0])

        my_prediction = model_Fraud.predict(df)
        message='FD'
        return render_template('single_predict.html',prediction_FD=my_prediction,message=message)

@app.route('/predict_LA',methods=['GET','POST'])
@cross_origin()
def predict_LA():
    if request.method == 'POST':
        Age = float(request.form.get("Age", False))
        Experience = float(request.form.get("Experience",False))
        Income = float(request.form.get("Income",False))
        Family = float(request.form.get("Family",False))
        CCAvg = float(request.form.get("CCAvg",False))

        Education = request.form.get("Education", False)
        if (Education == 'Undergrad'):
            Education = 0
        elif (Education == "Graduate"):
            Education = 1
        elif (Education == "Professional"):
            Education= 2

        Mortgage = float(request.form.get("Mortgage", False))

        SecuritiesAccount = request.form.get("SecuritiesAccount", False)
        if (SecuritiesAccount == 'Yes'):
            SecuritiesAccount = 1
        elif (SecuritiesAccount == "No"):
            SecuritiesAccount = 0

        CDAccount = request.form.get("CDAccount", False)
        if (CDAccount == 'Yes'):
            CDAccount = 1
        elif (CDAccount == "No"):
            CDAccount = 0

        Online = request.form.get("Online", False)
        if (Online == 'Yes'):
            Online = 1
        elif (Online == "No"):
            Online = 0

        CreditCard = request.form.get("CreditCard", False)
        if (CreditCard == 'Yes'):
            CreditCard = 1
        elif (CreditCard == "No"):
            CreditCard = 0

        LA_prediction = model_LA.predict(ss_LA.transform([[Age,Experience,Income,Family,CCAvg
                                                         ,Education,Mortgage,SecuritiesAccount,CDAccount,Online,CreditCard]]))
        message='LA'
        return render_template('single_predict.html',prediction_LA=LA_prediction,message=message)

@app.route('/predict_LE',methods=['GET','POST'])
@cross_origin()
def predict_LE():
    if request.method == 'POST':
        CurrentLoanAmount = float(request.form.get("CurrentLoanAmount",False))
        CreditScore = float(request.form.get("CreditScore",False))
        AnnualIncome = float(request.form.get("AnnualIncome",False))
        Yearsincurrentjob = float(request.form.get("Yearsincurrentjob",False))
        MonthlyDebt = float(request.form.get("MonthlyDebt",False))
        YearsofCreditHistory = float(request.form.get("YearsofCreditHistory",False))
        Monthssincelastdelinquent = float(request.form.get("Monthssincelastdelinquent",False))
        NumberofOpenAccounts = float(request.form.get("NumberofOpenAccounts",False))
        NumberofCreditProblems = float(request.form.get("NumberofCreditProblems",False))
        CurrentCreditBalance = float(request.form.get("CurrentCreditBalance",False))
        MaximumOpenCredit = float(request.form.get("MaximumOpenCredit",False))
        Term_LongTerm = float(request.form.get("Term_LongTerm",False))

        df = pd.DataFrame({
            "Monthssincelastdelinquent": Monthssincelastdelinquent, "MonthlyDebt": MonthlyDebt,
            "AnnualIncome": AnnualIncome, "CurrentCreditBalance": CurrentCreditBalance,
            "MaximumOpenCredit": MaximumOpenCredit, "CreditScore": CreditScore, "Yearsincurrentjob": Yearsincurrentjob,
            "Term_LongTerm": Term_LongTerm, "YearsofCreditHistory": YearsofCreditHistory,
            "NumberofOpenAccounts": NumberofOpenAccounts,
            "NumberofCreditProblems": NumberofCreditProblems, "CurrentLoanAmount": CurrentLoanAmount}, index=[0])

        LE_prediction = model_LE.predict(df)
        message='LE'
        return render_template('single_predict.html', prediction_LE=LE_prediction,message=message)

@app.route('/predict_MA',methods=['GET','POST'])
@cross_origin()
def predict_MA():
    if request.method == 'POST':
        PostCode = float(request.form.get("PostCode", False))
        Qtr = float(request.form.get("Qtr", False))
        Unit = float(request.form.get("Unit", False))

        df = pd.DataFrame({ "PostCode":PostCode, "Qtr":Qtr, "Unit":Unit},index=[0])
        MA_prediction = model_MA.predict(df)
        message='MA'
        return render_template('single_predict.html',prediction_MA=MA_prediction,message=message)

@app.route('/predict_LR',methods=['GET','POST'])
@cross_origin()
def predict_LR():
    if request.method == 'POST':
        Loan_Amount = float(request.form.get("LoanAmount",False))
        Term = float(request.form.get("Loan_Amount_Term",False))
        Interest_Rate = float(request.form.get("Interest_Rate",False))
        Employment_Years = float(request.form.get("Employment_Years",False))
        Annual_Income = float(request.form.get("Annual_Income",False))
        Debt_to_Income = float(request.form.get("Debt_to_Income",False))
        Delinquent_2yr = float(request.form.get("Delinquent_2yr",False))
        Revolving_Cr_Util = float(request.form.get("Revolving_Cr_Util",False))
        Total_Accounts = float(request.form.get("Total_Accounts",False))
        Longest_Credit_Length = float(request.form.get("Longest_Credit_Length",False))
        
        Home_Ownership = request.form.get("Home_Ownership",False)
        if Home_Ownership == 'RENT':
            Home_Ownership = 5
        elif Home_Ownership == 'OWN':
            Home_Ownership = 4
        elif Home_Ownership == 'MORTGAGE':
            Home_Ownership = 1
        elif Home_Ownership == 'OTHER':
            Home_Ownership = 3
        elif Home_Ownership == 'NONE':
            Home_Ownership = 2
        elif Home_Ownership == 'ANY':
            Home_Ownership = 0

        Verification_Status = request.form.get("Verification_Status",False)
        if Verification_Status == 'VERIFIED - income':
            Verification_Status = 1
        elif Verification_Status == 'VER' \
                                    'IFIED - income source':
            Verification_Status = 2
        elif Verification_Status == 'not verified':
            Verification_Status = 0
              
        Loan_Purpose = request.form.get("Loan_Purpose",False)
        if Loan_Purpose == 'credit_card':
            Loan_Purpose = 1
        elif Loan_Purpose =='car':
            Loan_Purpose = 0
        elif Loan_Purpose == 'small_business':
            Loan_Purpose = 11
        elif Loan_Purpose == 'other':
            Loan_Purpose = 9
        elif Loan_Purpose == 'wedding':
            Loan_Purpose = 13
        elif Loan_Purpose == 'debt_consolidation':
            Loan_Purpose = 2
        elif Loan_Purpose == 'home_improvement':
            Loan_Purpose = 4
        elif Loan_Purpose == 'major_purchase':
            Loan_Purpose = 6
        elif Loan_Purpose == 'medical':
            Loan_Purpose = 7
        elif Loan_Purpose == 'moving':
            Loan_Purpose = 8
        elif Loan_Purpose == 'renewable_energy':
            Loan_Purpose = 10
        elif Loan_Purpose == 'vacation':
            Loan_Purpose = 12
        elif Loan_Purpose == 'house':
            Loan_Purpose = 5
        elif Loan_Purpose == 'educational':
            Loan_Purpose = 3

        State = request.form.get("State",False)
        if State == 'AK':
            State = 0
        elif State == 'AL':
            State = 1
        elif State == 'AR':
            State = 2
        elif State == 'AZ':
            State = 3
        elif State == 'CA':
            State = 4
        elif State == 'CO':
            State = 5
        elif State == 'CT':
            State = 6
        elif State == 'DC':
            State = 7
        elif State == 'DE':
            State = 8
        elif State == 'FL':
            State = 9
        elif State == 'GA':
            State = 10
        elif State == 'HI':
            State = 11
        elif State == 'IA':
            State = 12
        elif State == 'ID':
            State = 13
        elif State == 'IL':
            State = 14
        elif State == 'IN':
            State = 15
        elif State == 'KS':
            State = 16
        elif State == 'KY':
            State = 17
        elif State == 'LA':
            State = 18
        elif State == 'MA':
            State = 19
        elif State == 'MD':
            State = 20
        elif State == 'ME':
            State = 21
        elif State == 'MI':
            State = 22
        elif State == 'MN':
            State = 23
        elif State == 'MO':
            State = 24
        elif State == 'MS':
            State = 25
        elif State == 'MT':
            State = 26
        elif State == 'NC':
            State = 27
        elif State == 'NE':
            State = 28
        elif State == 'NH':
            State = 29
        elif State == 'NJ':
            State = 30
        elif State == 'NM':
            State = 31
        elif State == 'NV':
            State = 32
        elif State == 'NY':
            State = 33
        elif State == 'OH':
            State = 34
        elif State == 'OK':
            State = 3
        elif State == 'OR':
            State = 36
        elif State == 'PA':
            State = 37
        elif State == 'RI':
            State = 38
        elif State == 'SC':
            State = 39
        elif State == 'SD':
            State = 40
        elif State == 'TN':
            State = 41
        elif State == 'TX':
            State = 42
        elif State == 'UT':
            State = 43
        elif State == 'VA':
            State = 44
        elif State == 'VT':
            State = 45
        elif State == 'WA':
            State = 46
        elif State == 'WI':
            State = 47
        elif State == 'WV':
            State = 48
        elif State == 'WY':
            State = 49

        LR_prediction = model_LR.predict([[Loan_Amount, Term, Interest_Rate, Employment_Years,
        Home_Ownership, Annual_Income, Verification_Status,
        Loan_Purpose, State, Debt_to_Income, Delinquent_2yr,
        Revolving_Cr_Util, Total_Accounts, Longest_Credit_Length]])
        message='Lrisk'
        return render_template('single_predict.html', prediction_Lrisk=LR_prediction,message=message)

@app.route('/predict_MS', methods=['GET', 'POST'])
@cross_origin()
def predict_MS():
    if request.method == 'POST':
        age = int(request.form.get("age", False))
        last_call_duration = float(request.form.get("last_call_duration", False))
        contacts_during_campaign = int(request.form.get("contacts_during_campaign", False))

        job = request.form.get("job", False)
        if (job == 'salaried'):
            salaried = 1
            self_employed = 0
            retired = 0
            unemployed = 0
            j_unknown = 0
            student = 0
        elif (job == 'self_employed'):
            salaried = 0
            self_employed = 1
            retired = 0
            unemployed = 0
            j_unknown = 0
            student = 0
        elif (job == 'retired'):
            salaried = 0
            self_employed = 0
            retired = 1
            unemployed = 0
            j_unknown = 0
            student = 0
        elif (job == 'unemployed'):
            salaried = 0
            self_employed = 0
            retired = 0
            unemployed = 1
            j_unknown = 0
            student = 0
        elif (job == 'unknown'):
            salaried = 0
            self_employed = 0
            retired = 0
            unemployed = 0
            j_unknown = 1
            student = 0
        elif (job == 'student'):
            salaried = 0
            self_employed = 0
            retired = 0
            unemployed = 0
            j_unknown = 0
            student = 1

        education = request.form.get("education", False)
        if (education == 'primary'):
            primary = 1
            secondary = 0
            tertiary = 0
            e_unknown = 0
            illiterate = 0
        elif (education == 'secondary'):
            primary = 0
            secondary = 1
            tertiary = 0
            e_unknown = 0
            illiterate = 0
        elif (education == 'tertiary'):
            primary = 0
            secondary = 0
            tertiary = 1
            e_unknown = 0
            illiterate = 0
        elif (education == 'illiterate'):
            primary = 0
            secondary = 0
            tertiary = 0
            e_unknown = 0
            illiterate = 1
        elif (education == 'e_unknown'):
            primary = 0
            secondary = 0
            tertiary = 0
            e_unknown = 1
            illiterate =0

        credit_default = request.form.get("credit_default", False)
        if (credit_default == 'c_no'):
            c_no= 1
            c_unknown = 0
            c_yes = 0
        elif (credit_default == 'c_unknown'):
            c_no = 0
            c_unknown = 1
            c_yes = 0
        elif (credit_default == 'c_yes') :
            c_no = 0
            c_unknown = 0
            c_yes = 1

        home_loan = request.form.get("home_loan", False)
        if (home_loan  == 'h_no'):
            h_no = 1
            h_yes = 0
            h_unknown = 0
        elif (home_loan == 'h_yes'):
            h_no = 0
            h_yes = 1
            h_unknown = 0
        elif (home_loan == 'h_unknown'):
            h_no = 0
            h_yes = 0
            h_unknown = 1

        personal_loan = request.form.get("personal_loan", False)
        if (personal_loan == 'p_no'):
            p_no = 1
            p_yes = 0
            p_unknown = 0
        elif (personal_loan == 'p_yes'):
            p_no = 0
            p_yes = 1
            p_unknown = 0
        elif (personal_loan == 'p_unknown'):
            p_no = 0
            p_yes = 0
            p_unknown = 1

        last_contacted_month = request.form.get("last_contacted_month", False)
        if (last_contacted_month == 'may_aug'):
            may_aug = 1
            sep_dec = 0
            jan_april = 0
        elif(last_contacted_month == 'sep_dec'):
            may_aug = 0
            sep_dec = 1
            jan_april = 0
        elif (last_contacted_month == 'jan_april'):
            may_aug = 0
            sep_dec = 0
            jan_april = 1

        day_of_week = request.form.get("day_of_week", False)
        if (day_of_week == 'mon'):
            mon = 1
            tue = 0
            wed = 0
            thu = 0
            fri = 0
        elif (day_of_week == 'tue'):
            mon = 0
            tue = 1
            wed = 0
            thu = 0
            fri = 0
        elif (day_of_week == 'wed'):
            mon = 0
            tue = 0
            wed = 1
            thu = 0
            fri = 0
        elif (day_of_week == 'thur'):
            mon = 0
            tue = 0
            wed = 0
            thu = 1
            fri = 0
        elif (day_of_week == 'fri'):
            mon = 0
            tue = 0
            wed = 0
            thu = 0
            fri = 1

        days_passed = request.form.get("days_passed", False)
        if (days_passed == 'never_contacted'):
            never_contacted = 1
            recent = 0
        elif(days_passed == 'recent'):
            never_contacted = 0
            recent = 1

        marital = request.form.get("marital", False)
        if (marital == 'married'):
            marry = 1
            single = 0
            divorced = 0
            m_unknown = 0
        elif (marital == 'single'):
            marry = 0
            single = 1
            divorced = 0
            m_unknown = 0
        elif (marital == 'divorced'):
            marry = 0
            single = 0
            divorced = 1
            m_unknown = 0
        elif (marital == 'unknown'):
            marry = 0
            single = 0
            divorced = 0
            m_unknown = 1

        cbc = request.form.get("cbc", False)
        if(cbc == "zero"):
            cbc_zero = 1
            cbc_less_than_ten = 0
        elif(cbc == 'less_than_10'):
            cbc_zero = 0
            cbc_less_than_ten=1

        MS_prediction = model_MS.predict([[age,last_call_duration,contacts_during_campaign,salaried,self_employed,retired,j_unknown,\
                                           student,unemployed,marry,single,divorced,m_unknown,primary,secondary,tertiary,e_unknown,illiterate,\
                                           c_no,c_yes, c_unknown,h_yes,h_no,h_unknown,p_no,p_yes,p_unknown, may_aug,sep_dec,jan_april,\
                                           mon,tue,wed,thu, fri,never_contacted,recent,cbc_zero,cbc_less_than_ten]])
        message='MS'
        return render_template('single_predict.html', prediction_MS=MS_prediction,message=message)

# routes for training data graph (button defined in main_dashboard.html)
@app.route('/FD_data_graph',methods = ['GET','POST'])
@cross_origin()
def FD_graph():
    if request.method == 'POST':
        return render_template('FD_data_graph.html')

@app.route('/LA_data_graph',methods = ['GET','POST'])
@cross_origin()
def LA_graph():
    if request.method == 'POST':
        return render_template('LA_data_graph.html')

@app.route('/Lrisk_data_graph',methods = ['GET','POST'])
@cross_origin()
def Lrisk_graph():
    if request.method == 'POST':
        return render_template('Lrisk_data_graph.html')

@app.route('/LE_data_graph',methods = ['GET','POST'])
@cross_origin()
def LE_graph():
    if request.method == 'POST':
        return render_template('LE_data_graph.html')

@app.route('/MA_data_graph',methods = ['GET','POST'])
@cross_origin()
def MA_graph():
    if request.method == 'POST':
        return render_template('MA_data_graph.html')

@app.route('/MS_data_graph',methods = ['GET','POST'])
@cross_origin()
def MS_graph():
    if request.method == 'POST':
        return render_template('MS_data_graph.html')

# route for top left home button (defined in sub html pages)
@app.route('/home1',methods = ['GET','POST'])
@cross_origin()
def home1():
    if request.method=='POST':
        return redirect(url_for('home'))

# show graph of bulk output data (defined in bulk_output_file.html)
@app.route('/show_graph',methods=['GET','POST'])
@cross_origin()
def show_graph():
    try:
        if request.method=='POST':
            graph_data =pd.read_csv(r'graph_input_files\graph_data.csv')
            prof = ProfileReport(graph_data)
            prof.to_file(output_file=r'templates\bulk_graph_output.html')
            return render_template('bulk_graph_output.html')
    except Exception as e:
        raise e

# download option for bulk output data (defined in bulk_output_file.html) ..
@app.route("/Down_Bulk_File",methods=['GET'])
@cross_origin()
def download_file():
    table = bulk_predict()
    response = make_response(table)
    response.headers['Content-Disposition'] = 'attachment; filename=report.csv'
    response.headers['Content-type'] = "text/csv"
    return response

# =================================================================================================================
if __name__ == '__main__':
    # To run on web ..
    ##app.run(host='0.0.0.0',port=8080)
    # To run locally ..
    app.run(host='0.0.0.0',debug=True)
