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
Fraud detection using mllib in spark. **Random Forest** and **xgboost** were applied on big data (7GB).
The Pyspark was used along with the data from TalkingData a Chinese company for fraud detection. Two ML algorithms including Random Forest and Xgboost were applied. This is a supervised classification problem and the objective is to predict whether a user will download an app after clicking a mobile app advertisement or Not.

### Project structure
---
The project structure provided at the following.
```
Fraud Detection Challenge using pyspark
Fraud_Detection_Pyspark/
├── assets/
│   ├── .ipynb_checkpoints/
│   ├── Docker.PNG
│   ├── Medi.PNG
│   ├── Medi2.PNG
│   └── tenor.gif
├── codes/
│   ├── .ipynb_checkpoints/
│   ├── Part 01 - Read data and EDA.ipynb
│   ├── RF_FraudDetection.ipynb
│   └── XGB_FraudDetection.ipynb
├── datasets/
│   └── train_sample.csv
├── Figures/
│   ├── All_distribution_plots.png
│   ├── All_distribution_plots_1.png
│   ├── All_distribution_plots_1_log.png
│   ├── All_distribution_plots_2_sea.png
│   ├── All_distribution_plots_oversampled.png
│   ├── All_distribution_plots_oversampled_transfered.png
│   ├── All_distribution_plots_transfered.png
│   ├── EDA_corr_plot.png
│   ├── parallel_categories.png
│   ├── parallel_categories2.png
│   ├── parallel_categories_oversampled.png
│   ├── parallel_categories_oversampled2.png
│   ├── parallel_categories_transfered_oversampled.png
│   ├── parallel_categories_transfered_oversampled2.png
│   ├── parallel_coordinates.png
│   └── plots_3.png
├── LICENSE
├── New folder/
├── README.md
└── Respo_folder_tree.ipynb
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
A pyspark session initiated and the data was read using read.csv as follows.

```python
data = spark.read.csv('train.csv', inferSchema=True, header=True)
data.show(5)
```
The first five rows of data are seen at the following table.
```
|    ip|app|device| os|channel|         click_time|attributed_time|is_attributed|
|------|---|------|---|-------|-------------------|---------------|-------------|
| 83230|  3|     1| 13|    379|2017-11-06 14:32:21|           null|            0|
| 17357|  3|     1| 19|    379|2017-11-06 14:33:34|           null|            0|
| 35810|  3|     1| 13|    379|2017-11-06 14:34:12|           null|            0|
| 45745| 14|     1| 13|    478|2017-11-06 14:34:52|           null|            0|
|161007|  3|     1| 13|    379|2017-11-06 14:35:08|           null|            0|
```

### Step 3 - Data cleaning and EDA
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
And count the distinct values as follow...
```python
data.agg(*(countDistinct(col(c)).alias(c) for c in data.columns)).show()
```
```
|    ip|app|device| os|channel|click_time|attributed_time|is_attributed|
|------|---|------|---|-------|----------|---------------|-------------|
|277396|706|  3475|800|    202|    259620|         182057|            2|
```
The column "is_attributed" is the goal here. Let's see whether the data is balanced or not.

```python
data.groupBy('is_attributed').count().show()
```
```
|is_attributed|    count|
|-------------|---------|
|            1|   456846|
|            0|184447044|
```
As it is seen the data is a very imbalance with just 0.25% value of 1.

### Step 4 - Oversampling, dropping some columns, splitting, vectorizing, and training.
The calculated ratio above is used for oversampling the 1 and achieve balance dataset before using in the training.
```

|is_attributed|    count|
|-------------|---------|
|            1|184108938|
|            0|184447044|

```
The data later is combined and fed into the Ml algorithm.
For the sack of simplicity, two columns are dropped including the click time and attributed time.
The performance of the model was evaluated.

### Step 5 - Applying trained model on test data set.
The test data set was prepared and similarity two columns dropped and vectorized. The trained model was applied and the results were submitted.
```python
data_to_submit.groupBy('is_attributed').count().show()
```
Lets before submitting the results for the Random Forest algorithm, look inside and see the model performance.
```
Random Forest results:
|is_attributed|   count|
|-------------|--------|
|          0.0|15921222|
|          1.0| 2869247|

XGBoost results:
|is_attributed|   count|
|-------------|--------|
|          0.0|17946533|
|          1.0|  843936|
+-------------+--------+
```
### Recommendation and next steps
* Hyperparameter tuning:
The hyperparameters have a big impact on the output of ML algorithms performance. It is suggested to run the algorithm on a small sample size with different hyper-parameters values and compare and find the best set. 
Another approach is to use powerful searching algorithm like the DE algorithm to search the Hyperparameter domain and to find the best available set. This is have been done in python [here](https://github.com/Atashnezhad/Fraud_detection_and_XGBoost). The results were very promising with an accuracy of 89% on the test data set.
* Data feathering: 
It is suggested to feather new columns using available data. Feeding more input parameters can be helpful if the new feathered data are prepared wisely.



### Do you wanna run codes without Hassle? Try Docker Technology then!
 

<p align="center">
  <img width="1000" src="assets/Docker.PNG" >
</p>



  
Please check out my docker hub account [here](https://hub.docker.com/repository/docker/atashnezhad/ml_in_spark).
To have this project you may simply pull the image from docker hub and run the codes yourself.

### Don't know how to use the Docker technology? See my tutorial on Medium.

<p align="left">
  <img width="500" src="assets/Medi2.PNG" >
</p>

Please check my toturial on Running Python code in Jupyter notebook on Docker on [Medium](https://medium.com/@atashnezhad1/in-this-tutorial-we-will-learn-the-very-basics-of-running-the-jupyter-notebook-using-docker-9b347c9058d9).


[Gif reference](https://tenor.com/view/phishing-phisher-hacker-security-gif-16575067).

