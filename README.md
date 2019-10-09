# **Predictive Modeling for Direct Marketing**
A bank is marketing Certificates of Deposit (CD) to a pool of 1,000 potential customers. The cost of each contact is $10. The NPV to the bank of a customer buying a CD is $50. Clearly, it is not profitable to contact all the potential customers. So, instead the bank employs a predictive model to predict whether or not a potential customer is a likely buyer.
Our objective is to build a predictive model that maximizes profit for the project. 

## **Data description**   
The data set has 16 features and a class variable for a total of 17 columns that are described in the table below. It has 4,521 observations.

| Feature | Description | 
| ------ | :--------------------: |  
| age     | Age of customer in years (integer)    | 
| job     | Job category, one of 12 possible values, (string)    | 
| marital | Marital status, one of 3 possible values, (string)   | 
| education  | Level of education, one of 4 possible values, (string)    | 
| default | Whether the customer has previously defaulted on a loan (yes or no)  | 
| balance | Average daily balance (numeric)  | 
| housing | Whether the customer has a mortgage with the bank (yes or no)  | 
| loan | Whether the customer has a revolving loan credit with the bank (yes or no)   | 
| day |  Date of last contact | 
| month | Month of last contact  | 
| contact | Method of last contact, one of three values (string)   | 
| duration | Duration of last contact, in minutes, (integer)   | 
| campaign | Campaign identifier  | 
| pdays | Days to prior contact  | 
| previous | Number of previous contacts  | 
| poutcome | Outcome of previous  | 
| y |  Whether the customer purchased a CD or not (yes, or no), the class variable | 

