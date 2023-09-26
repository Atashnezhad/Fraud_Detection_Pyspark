from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct
from pyspark.sql.functions import col, explode, array, lit
# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName('FraudTreeMethods').getOrCreate()


if __name__ == "__main__":
    # Load and parse the data file, converting it to a DataFrame.
    #data = sqlContext.sql("SELECT * FROM fraud_train_sample")
    data = spark.read.csv('train.csv', inferSchema=True, header=True)
    data.show(5)
