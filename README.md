<p align="center">
  <img width="460" height="300" src="assets/tenor.gif" >
</p>


# Fraud Detection Challenge using pyspark

This is a classification problem. The ML algorithm should detect the fraud clicks, taking the click time, IP address, etc into account. The provided data is very unbalanced with less than 1% fraud clicks.

Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest
mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user’s click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

### Project Statement
---
Fraud detection using mllib in spark. Random Forest and xgboost were applied on big data (7GB).
The Pyspark was used along with the data from TalkingData a Chinese company for fraud detection. Two ML algorithms including Random Forest and Xgboost were applied. This is a supervised classification problem and the objective is to predict whether a user will download an app after clicking a mobile app advertisement or Not.

### Project structure
---
The project structure provided at the following.
```
project-global_warming_NLP
    
|__ codes/
|   |__ RF_FraudDetection.ipynb
|   |__ XGB_FraudDetection.ipynb   
|__ datasets/
|__ plots/
|__ README.md
```

### Step 1 - Data descriptions
---
The data in this study was provided by TalkingData company at the following [Link](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data).
The data are in csv format and incudes the train, train_sample, test, and sample_Submission which is used to show the format of submitting csv file.

train.csv - the training set
train_sample.csv - 100,000 randomly-selected rows of training data, to inspect data before downloading full set
test.csv - the test set
sampleSubmission.csv - a sample submission file in the correct format
UPDATE: test_supplement.csv - This is a larger test set that was unintentionally released at the start of the competition. It is not necessary to use this data, but it is permitted to do so. The official test data is a subset of this data.
Data fields
Each row of the training data contains a click record, with the following features.
```
root
 |-- ip: integer (nullable = true)
 |-- app: integer (nullable = true)
 |-- device: integer (nullable = true)
 |-- os: integer (nullable = true)
 |-- channel: integer (nullable = true)
 |-- is_attributed: integer (nullable = true)
 |-- features: vector (nullable = true)
```

* ip: ip address of click.
* app: app id for marketing.
* device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
* os: os version id of user mobile phone
* channel: channel id of mobile ad publisher
* click_time: timestamp of click (UTC)
* attributed_time: if user download the app for after clicking an ad, this is the time of the app download
* is_attributed: the target that is to be predicted, indicating the app was downloaded

Note that ip, app, device, os, and channel are encoded.

The test data is similar, with the following differences:

click_id: reference for making predictions
is_attributed: not included


### Step 2 - Data descriptions
A pyspark session initiated and the data was read using read.csv as follow.

```python
data = spark.read.csv('train.csv', inferSchema=True, header=True)
data.show(5)
data.show(5)
```
The first five rows of data are seen at the following table.
```
|    ip|app|device| os|channel|         click_time|attributed_time|is_attributed|
|------|---|------|---|-------|-------------------|---------------|-------------|
| 87540| 12|     1| 13|    497|2017-11-07 09:30:38|           null|            0|
|105560| 25|     1| 17|    259|2017-11-07 13:40:27|           null|            0|
|101424| 12|     1| 19|    212|2017-11-07 18:05:24|           null|            0|
| 94584| 13|     1| 13|    477|2017-11-07 04:58:08|           null|            0|
| 68413| 12|     1|  1|    178|2017-11-09 09:00:09|           null|            0|
```

### Step 3 - Data cleaning
In this study for sack of simplicity the two columns including the click_time and attributed_time are deleted from data.
```python
data = data.drop('click_time','attributed_time')
data.show(3)
```
```
|    ip|app|device| os|channel|is_attributed|
|------|---|------|---|-------|-------------|
| 87540| 12|     1| 13|    497|            0|
|105560| 25|     1| 17|    259|            0|
|101424| 12|     1| 19|    212|            0|
```










