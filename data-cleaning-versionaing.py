import pandas as pd
import boto3
from datetime import date

path=pd.read_csv("Mlops_house_predication_raw_data.csv")

# data=pd.read_csv(path)
df=pd.DataFrame(path)

print("===============Before Cleaning=============")
print(df.isnull().sum())
print(f"Shape Before : {df.shape}")
print()

df_clean=df.dropna()
print("===============After Cleaning===============")
print(df_clean.isnull().sum())
print(f"Shape After : {df_clean.shape}")

clean_path="Mlops_house_predication_raw_data_clean_v1.csv"
df_clean.to_csv(clean_path,index=False)



s3=boto3.client('s3')
BUCKET="dhurandhar-bucket-750"
def upload_proccessed_data(local_path):
    key=f"proccessed/{date.today()}/Mlops_house_predication_raw_data_clean_v1.csv"
    s3.upload_file(local_path,BUCKET,key)
    print(f"\nUploaded to s3://{BUCKET}/{key}")
    return key

upload_proccessed_data(clean_path)