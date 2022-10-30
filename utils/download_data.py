# Importing packages
import boto3 



# def download_data(bucket: str,destination:str) -> None:
#     ''' Here we are downloading our dataset from the given bucket 

#         args: 
#             bucket(str): location of where our dataset is. 
#             destination(str): 
#     '''

if __name__ == "__main__":

    s3 = boto3.resource('s3')
    bucket = s3.Bucket('mle9-spotify-wakeword-data')
    
    for obj in bucket.objects.filter(Prefix="train"):
        print(obj.key)
    # for bucket in s3.buckets.all():
    #     print(bucket.name)