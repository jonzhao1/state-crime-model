# Logistic regression model to predict crime
# Run in Amazon EMR cluster 

$ pyspark

sc.setLogLevel("ERROR")


# Import some functions we will need later on
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
import numpy as np 


# Set up the path to the final dataset in S3 bucket
bucket = 'big-data-project-1/'
filename = 'datacommons_api.csv'
file_path = 's3a://' + bucket + filename

# Create a Spark Dataframe from the file on S3
sdf = spark.read.csv(file_path, header=True, inferSchema=True)

# create column for crime per 100,000 people 
# floor function rounds the calculation down
sdf = sdf.withColumn('Crime_Per_Hundred_Thousand', floor(sdf.Crime / sdf.Population * 100000))

# show some data
sdf_select = sdf.select('State','Year','Crime','Population','Crime_Per_Hundred_Thousand').show(10)

# calculate national average number of crimes per 100,000 people for each year
average_crime_sdf = sdf.groupBy('Year').agg(round(mean('Crime_Per_Hundred_Thousand')).alias('average_crime'))

# Show the average crime
average_crime_sdf.show(10)

# Join the average crime dataframe back into the original dataframe sdf
sdf = sdf.join(average_crime_sdf, "Year")

# Show some of the new, joined data
sdf.select('Year','State','Population','Crime','Crime_Per_Hundred_Thousand','average_crime').show(10)


# Create the label, =1 if Crime_per_Hundred_Thousand > average_crime, =0 otherwise
sdf = sdf.withColumn("highercrime", when(sdf.Crime_Per_Hundred_Thousand > sdf.average_crime, 1.0).otherwise(0.0))

# Show some of the data with the label
sdf_select = sdf.select('Year','State','Population','Crime','Crime_per_Hundred_Thousand','average_crime','highercrime').show(10)



# Prepare data for modeling

# Feauture Engineering

# Data types of columns
sdf.printSchema()  

# Trying to encode the features: age, population, income using MinMaxScaler to scale the values to a 0.0 to 1.0 range and then putting them in the vector assembler makes them too large and PySpark runs out of memory
# For example, if you have a population of 4799277, then the vector will have at least 4799277 elements. Combine that with the Income and each vector will have millions of elements
# Instead will treat age, population, income as double and without encoding them, place into vector assembler directly as features

# change Income to double
sdf = sdf.withColumn("Income", sdf.Income.cast("double"))


# the feature State is a string 


# StringIndexer
indexer = StringIndexer(inputCols=['State'], outputCols=['stateIndex'])


# OneHotEncoder
encoder = OneHotEncoder(inputCols=['stateIndex'], outputCols=['stateVector'], dropLast=False)


# Vector Assembler
assembler = VectorAssembler(inputCols=['stateVector','Population','Income', 'Age'],
outputCol="features")


# Create pipeline
crime_pipe = Pipeline(stages=[indexer, encoder, assembler])

# Call .fit to transform the data
transformed_sdf = crime_pipe.fit(sdf).transform(sdf)

# Review the transformed features
transformed_sdf.select('State','stateVector','Year','Population','Income','Age','Crime_per_Hundred_Thousand','average_crime','highercrime','features').show(20, truncate=False)

# Split data
train, test = transformed_sdf.randomSplit([.7, .3], seed=3456)

# LogisticRegression
lr = LogisticRegression(labelCol="highercrime")

# Fit the model
model = lr.fit(train)

# Show model coefficients and intercept
print("Coefficients: ", model.coefficients)
print("Intercept: ", model.intercept)

# Test the model on the test data
test_results = model.transform(test)

# Test Results

# Show the test results
test_results.select('State','Year','Population','Income','Age','Crime_per_Hundred_Thousand','average_crime','rawPrediction','probability','prediction','highercrime').show(truncate=False)

# Show the confusion matrix
test_results.groupby('highercrime').pivot('prediction').count().show()


# Model Validation


# Create a BinaryClassificationEvaluator to evaluate how well the model works 
evaluator = evals.BinaryClassificationEvaluator(labelCol="highercrime", metricName="areaUnderROC")

# Create the parameter grid (empty for now) 
grid = tune.ParamGridBuilder().build()

# Create the CrossValidator 
cv = tune.CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3, seed=789)

# Use the CrossValidator to Fit the train data 
cv = cv.fit(train)

# Show the average performance over the three folds 
cv.avgMetrics


# Evaluate the test data using the cross-validator model 
# Reminder: We used Area Under the Curve 
evaluator.evaluate(cv.transform(test))


# Tuning
# Explore a range of Hyperparameters, carry out multiple splits and then see which parameters lead to the best model performance

# Create a grid to hold hyperparameters
# Logistic Regression threshold from 0 to 0.1 in .01 increments

grid = tune.ParamGridBuilder() 
grid = grid.addGrid(lr.regParam, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
grid = grid.addGrid(lr.elasticNetParam, [0, 1]) 

# Build the grid 
grid = grid.build() 
print('Number of models to be tested: ', len(grid)) 

# Create the CrossValidator using the new hyperparameter grid 
cv = tune.CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator) 

# Call cv.fit() to create models with all of the combinations of parameters in the grid 
all_models = cv.fit(train) 

print("Average Metrics for Each model: ", all_models.avgMetrics)


# Get the best model

# Gather the metrics and parameters of the model with the best average metrics
hyperparams = all_models.getEstimatorParamMaps()[np.argmax(all_models.avgMetrics)]

# Print out the list of hyperparameters for the best model
for i in range(len(hyperparams.items())):
    print([x for x in hyperparams.items()][i])

#(Param(parent='LogisticRegression_daf2ca402a50', name='regParam', doc='regularization parameter (>= 0).'), 1.0)
#(Param(parent='LogisticRegression_daf2ca402a50', name='elasticNetParam', doc='the ElasticNet mixing parameter, in 
#range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'), 0.0)

# Choose the best model
bestModel = all_models.bestModel
print("Area under ROC curve:", bestModel.summary.areaUnderROC)    # Area under ROC curve: 0.99205912414498


# Test the best model on the test set

# Use the model 'bestModel' to predict the test set
test_results = bestModel.transform(test)

# Show the results
test_results.select('stateVector','Population','Income', 'Age', 'probability', 'prediction',
'highercrime').show(truncate=False)

# Evaluate the predictions. Area Under ROC curve
print(evaluator.evaluate(test_results))    # 0.9820261437908496 
