# Loop through entire Amazon S3 bucket and get descriptive statistics for every CSV object (also works for every CSV object in folders)

import boto3
import pandas as pd
bucket_name="big-data-project-1"   # Put your bucket name here
s3_client = boto3.client('s3', use_ssl=False)
s3_resource = boto3.resource('s3')

def get_statistics(filename):
    # Do something here to get the statistics about the current file
    df = pd.read_csv(file_path, low_memory=False)
    print("Filename:", filename)
    print("\nNumber Rows and Columns:", df.shape)
    print("\nColumn Names:", df.columns)
    print("\nData Types:", df.dtypes)
    print("\nNumber Missing Values:", df.isnull().sum())
    print("\nMissing Values:", df[df.isnull().any(axis=1)])
    # Retrieve Descriptive Statistics and save as csv in the S3 bucket
    stats_filename = filename + "_stats.csv"
    df.describe(include='all').to_csv(stats_filename)
    df2 = pd.read_csv(stats_filename)
    print("\nDescriptive Statistics:", df2)

# Loop through objects in bucket
for object in s3_client.list_objects(Bucket=bucket_name)['Contents']:
    filename = object['Key']
    if ".csv" in filename:
        print('Working on file name:', filename)
        # Create the full path to the file in the bucket
        file_path = "s3://" + bucket_name + "/" + filename
        # Call your function to analyze filename
        get_statistics(file_path)
