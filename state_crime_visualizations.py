# Visualizations of the data and prediction results
# Run in Amazon EMR cluster

# each time you create a new EMR cluster, have to install packages again
$ pip3 install s3fs
$ pip3 install matplotlib
$ pip3 install plotly-express
$ pip3 install kaleido

$ pyspark

sc.setLogLevel("ERROR")


# Import some functions we will need later on
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
import pandas as pd
import numpy as np 
import io
import s3fs
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import plotly.express as px


# Set up the path to the final dataset in S3 bucket
bucket = 'bucket-name/'   # put your bucket here
filename = 'dataset.csv'   # put the name of your dataset file here
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
sdf.select('State','Year','Crime','Population','Crime_Per_Hundred_Thousand','average_crime').show(10)


# Create the label, =1 if Crime_per_Hundred_Thousand > average_crime, =0 otherwise
sdf = sdf.withColumn("highercrime", when(sdf.Crime_Per_Hundred_Thousand > sdf.average_crime, 1.0).otherwise(0.0))


# (for visualization) Create the label, =1 if Crime_per_Hundred_Thousand > average_crime, =0 otherwise
sdf = sdf.withColumn("State Crime", when(sdf.Crime_Per_Hundred_Thousand > sdf.average_crime, 1.0).otherwise(0.0))


# Show some of the data with the label
sdf_select = sdf.select('State','Year','Crime','Population','Crime_per_Hundred_Thousand','average_crime','highercrime').show(10)



# Prepare data for modeling

# Feauture Engineering

# Data types of columns
sdf.printSchema()   

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


# Model Validation


# Create a BinaryClassificationEvaluator to evaluate how well the model works 
evaluator = evals.BinaryClassificationEvaluator(labelCol="highercrime", metricName="areaUnderROC")


# AOC of test results
print(evaluator.evaluate(test_results))    # 0.9871615312791783


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

(Param(parent='LogisticRegression_daf2ca402a50', name='regParam', doc='regularization parameter (>= 0).'), 1.0)
(Param(parent='LogisticRegression_daf2ca402a50', name='elasticNetParam', doc='the ElasticNet mixing parameter, in 
range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'), 0.0)

# Choose the best model
bestModel = all_models.bestModel





# Visualizations


# Show the confusion matrix
test_results.groupby('highercrime').pivot('prediction').count().show()


cm = test_results.groupby('highercrime').pivot('prediction').count().fillna(0).collect()

def calculate_precision_recall(cm):
    tn = cm[0][1]
    fp = cm[0][2]
    fn = cm[1][1]
    tp = cm[1][2]
    precision = tp / ( tp + fp )
    recall = tp / ( tp + fn )
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return accuracy, precision, recall, f1_score

print(calculate_precision_recall(cm))    # (0.04580152671755725, 0.05714285714285714, 0.06349206349206349, 0.06015037593984963)





# ROC CURVE

plt.figure(figsize=(6,6))
plt.plot([0, 1], [0, 1], 'r--')
x = bestModel.summary.roc.select('FPR').collect()
y = bestModel.summary.roc.select('TPR').collect()
plt.scatter(x, y)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")

# Save plot to S3
# Create a buffer to hold the figure
img_data = io.BytesIO()

# Write the figure to the buffer
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

# Connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://bucket-name/ROC_Curve.png', 'wb') as f:
    f.write(img_data.getbuffer())




# Coefficients

# Extract the coefficients on each of the variables
coeff = bestModel.coefficients.toArray().tolist()

# Loop through the features to extract the original column names. Store in the var_index dictionary
var_index = dict()
for variable_type in ['numeric', 'binary']:
    for variable in test_results.schema["features"].metadata["ml_attr"]["attrs"][variable_type]:
        print("Found variable:", variable)
        idx = variable['idx']
        name = variable['name']
        var_index[idx] = name # Add the name to the dictionary

# Loop through all of the variables found and print out the associated coefficients
for i in range(len(var_index)):
    print(i, var_index[i], coeff[i])




# Convert national average (from groupby) spark dataframe to pandas dataframe for line graphs
df2 = average_crime_sdf.toPandas()



# Line graph of change in national average crime by year

# Have to sort the Year column in order to have the correct line graph
df2_sorted = df2.sort_values('Year')

# Create line graph
plt.plot(df2_sorted['Year'], df2_sorted['average_crime'], linestyle='--', color='red', marker='o',label='Average National Crime')

# add title and labels
plt.title('United States Average National Crime by Year')
plt.xlabel('Year')
plt.ylabel('Average National Crime')

# add gridlines
plt.grid(True)

# Save plot to S3
# Create a buffer to hold the figure
img_data = io.BytesIO()

# Write the figure to the buffer
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

# Connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://bucket-name/line_national_crime.png', 'wb') as f:
    f.write(img_data.getbuffer())






# Bar graph combined with line graph of percent change in national average crime by year

# Have to sort the Year column in order to have the correct line graph
df2_sorted = df2.sort_values('Year')


# Calculate the percent change of crime with each increasing year
df2_sorted['crime_pct_change'] = df2_sorted['average_crime'].pct_change()


# Remove the first value (nan) of 2011 because there is no previous year in the dataset
df2_sorted = df2_sorted.dropna()


# Create line graph
plt.plot(df2_sorted['Year'], df2_sorted['crime_pct_change'], linestyle='--', color='red', marker='o',label='Percent Change of Average National Crime')

# Create bar graph
plt.bar(x=df2_sorted['Year'], height=df2_sorted['crime_pct_change'], width=0.4, color='blue', edgecolor='k')


# Set the y-axis tick labels to be shown as percentages with the percent symbol (multiplies the number by 100 and adds % symbol)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

# Invert the y-axis so the bars appear in the correct orientation
plt.gca().invert_yaxis()


# add title and labels
plt.title('United States Percent Change in Average National Crime by Year')
plt.xlabel('Year')
plt.ylabel('Percent Change in Average National Crime')

# add gridlines
plt.grid(True)

# Save plot to S3
# Create a buffer to hold the figure
img_data = io.BytesIO()

# Write the figure to the buffer
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

# Connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://bucket-name/line_bar_pct_change_national_crime.png', 'wb') as f:
    f.write(img_data.getbuffer())





# Convert spark dataframe to pandas dataframe for plotly visuals
df = sdf.toPandas()



# Map of U.S. for whether the average crime rate per 100,000 people for each state is greater than the national average for 2011

# create dataframe filtering values only from 2011
df2 = df[df.Year == 2011]     # filter the original dataframe to only contain rows where Year column is 2011
df2 = df2.rename({'State Crime': 'State Crime 2011'}, axis=1)

# Create map
fig = px.choropleth(df2,
                    locations='State_Abbreviation', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='State Crime 2011',
                    color_continuous_scale="viridis_r"
                    )

# Add titles
fig.update_layout(
      title_text = '2011 Crime Greater Than National Average by State',
      title_font_family="Times New Roman",
      title_font_size = 22,
      title_font_color="black", 
      title_x=0.45, 
         )


# Save plot to S3
# Create a buffer to hold the figure
img_data = io.BytesIO()

# Write the figure to the buffer
fig.write_image(img_data)
img_data.seek(0)

# Connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://bucket-name/map_national_crime_2011.png', 'wb') as f:
    f.write(img_data.getbuffer())




# Map of U.S. for whether the average crime rate per 100,000 people for each state is greater than the national average for 2019

# create dataframe filtering values only from 2019
df3 = df[df.Year == 2019]     # filter the original dataframe to only contain rows where Year column is 2019
df3 = df3.rename({'State Crime': 'State Crime 2019'}, axis=1)

# Create map
fig = px.choropleth(df3,
                    locations='State_Abbreviation', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='State Crime 2019',
                    color_continuous_scale="viridis_r" 
                    )

# Add titles
fig.update_layout(
      title_text = '2019 Crime Greater Than National Average by State',
      title_font_family="Times New Roman",
      title_font_size = 22,
      title_font_color="black", 
      title_x=0.45, 
         )


# Save plot to S3
# Create a buffer to hold the figure
img_data = io.BytesIO()

# Write the figure to the buffer
fig.write_image(img_data)
img_data.seek(0)

# Connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://bucket-name/map_national_crime_2019.png', 'wb') as f:
    f.write(img_data.getbuffer())






# Map of U.S. for the difference in Crime_Per_Hundred_Thousand and average_crime (national average crime per 100,000) in 2011

# create dataframe filtering values only from 2011
df4 = df[df.Year == 2011]     # filter the original dataframe to only contain rows where Year column is 2011
df4['Crime Difference 2011'] = df4['Crime_Per_Hundred_Thousand'] - df4['average_crime']


# Create map
fig = px.choropleth(df4,
                    locations='State_Abbreviation', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Crime Difference 2011',
                    color_continuous_scale="blues" 
                    )

# Add titles
fig.update_layout(
      title_text = '2011 Difference Between Crime and National Average by State',
      title_font_family="Times New Roman",
      title_font_size = 19,
      title_font_color="black", 
      title_x=0.45, 
         )


# Save plot to S3
# Create a buffer to hold the figure
img_data = io.BytesIO()

# Write the figure to the buffer
fig.write_image(img_data)
img_data.seek(0)

# Connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://bucket-name/map_crime_difference_2011.png', 'wb') as f:
    f.write(img_data.getbuffer())





# Map of U.S. for the difference in Crime_Per_Hundred_Thousand and average_crime (national average crime per 100,000) in 2019

# create dataframe filtering values only from 2019 
df5 = df[df.Year == 2019]     # filter the original dataframe to only contain rows where Year column is 2019
df5['Crime Difference 2019'] = df5['Crime_Per_Hundred_Thousand'] - df5['average_crime']


# Create map
fig = px.choropleth(df5,
                    locations='State_Abbreviation', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Crime Difference 2019',
                    color_continuous_scale="blues" 
                    )

# Add titles
fig.update_layout(
      title_text = '2019 Difference Between Crime and National Average by State',
      title_font_family="Times New Roman",
      title_font_size = 19,
      title_font_color="black", 
      title_x=0.45, 
         )


# Save plot to S3
# Create a buffer to hold the figure
img_data = io.BytesIO()

# Write the figure to the buffer
fig.write_image(img_data)
img_data.seek(0)

# Connect to the s3fs file system
s3 = s3fs.S3FileSystem(anon=False)
with s3.open('s3://bucket-name/map_crime_difference_2019.png', 'wb') as f:
    f.write(img_data.getbuffer())
