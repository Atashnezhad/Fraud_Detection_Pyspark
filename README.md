<p align="center">
  <img width="460" height="300" src="assets/tenor.gif" >
</p>


# Fraud Detection Challenge using pyspark

This is a classification problem. The ML algorithm should detect the fraud clicks, taking the click time, IP address, etc into account. The provided data is very unbalanced with less than 1% fraud clicks.

Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest
mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user’s click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

### Project Statement
 
Fraud detection using mllib in spark. Random Forest and xgboost were applied on big data (7GB).

The Pyspark was used along with the data from TalkingData Chinese company for fraud detection. Two ML algorithms including Random Forest and Xgboost were applied. This is a supervised classification problem.

the data divided to 70% and 30% for training and testing purposes. click time was dropped from data. accuracy was used as metric evaluation of the models.


### Project structure








