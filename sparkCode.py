from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc= SparkContext()
sqlContext = SQLContext(sc)
house_df = sqlContext.read.format('com.databricks.spark.csv').options(delimiter=";",header='true', inferschema='true').load('Training_dataset.csv')
house_df.take(1)

from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'], outputCol = 'features')
vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['features', 'quality'])

h_df = sqlContext.read.format('com.databricks.spark.csv').options(delimiter=";",header='true', inferschema='true').load('ValidationDataset.csv')
h_df.take(1)

vh_df = vectorAssembler.transform(h_df)
vh_df = vh_df.select(['features', 'quality'])

train_df = vhouse_df
test_df = vh_df

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol='quality', numTrees=777)
rf_model = rf.fit(train_df)

predictions = rf_model.transform(test_df)
predictions.select("prediction","quality","features").show()

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
#accuracy = evaluator.evaluate(predictions)
#print("Test Error = %g" % (1.0 - accuracy))

f1 = evaluator.evaluate(predictions, {evaluator.metricName: 'f1'})

print("F1 Score = %s" % f1)



