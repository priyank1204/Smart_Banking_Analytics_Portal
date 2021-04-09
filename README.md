# Smart Banking Analytics Portal


## Table of Content
  * [Overview](#overview)
  * [Installation](#installation)
  * [Dataset details](#Dataset-details)
  * [UI overview](#UI-overview)
  * [Technologies Used](#technologies-used)

### Overview
This project is an end to end machine learning project which is a banking portal that is build using different datasets and training them .In this portal, there are different models that are integrated into a User Interface . These models are Fraud Detection,Loan Repayment prediction, Loan eligibilty Prediction,Mortgage Analytics etc.  All these models are build using different datasets . ALso the user will be able to upload a file and get predictions for that using this user friendly portal.

## Installation 
For the portal to run, python packages will be required .This whole portal is build on python38 so the required packages are mentioned in a file named as requirements.txt from where we can download all the required libraries in a single go by just typing the following command in the command prompt after installing python
 
 ```bash
pip install -r requirements.txt
```

 ## Dataset details
 As this is an end to end banking portal,so there are more than one datasets that have been used in this. The dataset for this portal are collected from different repositeries  like kaggle ,UCI etc.
 The list of the datasets that i have used for this portal are as follows:
 
 1) Fraud Detection Banksim Dataset from Kaggle
 2) Home credit Default dataset
 3) Mortgage Dataset
 4) Marketing Dataset
 5) Loan Acceptance Dataset
 6) Loan Eligibility Dataset
 7) Loan Risk Dataset

## UI Overview
In this Banking portal,i have build an UI which is user friendly in all aspects. If a person from non-technical background wants to use the portal ,then they can also do so.In the UI i have added many functionalities as given below:

### Multi Purpose Portal:
As i have mentioned before that this project is and end to end banking portal which has many functionalities based on differet datasets so they are added in UI as well as shown below:

![2](https://user-images.githubusercontent.com/53222813/114068303-67a36580-98bb-11eb-8e92-6f1832fad90f.JPG)

![3](https://user-images.githubusercontent.com/53222813/114070678-ded9f900-98bd-11eb-88df-e767811deea3.JPG)

### Single Input:
If the user wants to predict only for a single record then he can enter the values for the same and get the prediction . Here in below i have shown a sample of single input where user can give input for these values . Here is Fraud Detection sample:

![4](https://user-images.githubusercontent.com/53222813/114068676-cc5ec000-98bb-11eb-82d6-44518923a773.JPG)    

### Bulk user prediction:
* Also there is an option of multiple input , if the user wants to predict for multiple reocrds then they can go with bulk prediction option . Here the user has to give a file input and then they can get the output for the same. 

* Here is also option of a report with visulization of the data by which the particular modle is trained , if user wants to get insights about the data then he can go with the visualization option.

* Performance metrics of the trained model are also displayed on the screen if in case user wants to know about the model performance. 

![5](https://user-images.githubusercontent.com/53222813/114072773-3d07db80-98c0-11eb-9d44-dee5dc677f4a.JPG)


### Output :
If the user goes with Bulk prediction then after submitting the file the output will be shown in this way as given below:

* After getting the output for the input file the user will get the predictions and along with that he can get graphs for the bulk output .

* Also the user can download the predictions in form of a csv file.

* Also to get the full insights about the bulk input data he can go with the option of full data visuals.  

![6](https://user-images.githubusercontent.com/53222813/114071940-50ff0d80-98bf-11eb-8c46-0a3e82f65027.JPG)

 
### Technologies Used

In this whole ploject we have used the following python libraries extensively

![flask](https://user-images.githubusercontent.com/53222813/91666223-b7959f00-eb18-11ea-9c27-46badbab0367.png)
![html](https://user-images.githubusercontent.com/53222813/91666224-b8c6cc00-eb18-11ea-9735-27ba3d0493e8.png)
![panda](https://user-images.githubusercontent.com/53222813/91666225-b95f6280-eb18-11ea-84cf-59e719287594.png)
![sklearn](https://user-images.githubusercontent.com/53222813/91666226-b95f6280-eb18-11ea-87e1-e26bc87f8ba8.png)
![download (1)](https://user-images.githubusercontent.com/53222813/91881104-a1195000-ec9e-11ea-9abd-6c16653603f9.png)
![download](https://user-images.githubusercontent.com/53222813/91881107-a37baa00-ec9e-11ea-8761-2cba0133f2b7.png)
![download](https://user-images.githubusercontent.com/53222813/91887750-cdd26500-eca8-11ea-83c4-2d5f999418e7.png)
