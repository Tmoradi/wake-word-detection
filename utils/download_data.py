# Importing packages
import boto3
import os
def download_data(Prefix:list[str]) -> None:
    ''' Here we are downloading our dataset from the given bucket 

        args: 
            bucket(list[str]): location of where our dataset is. 
            destination(list[str]):  
    ''' 
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('mle9-spotify-wakeword-data')
    
    for prefix in Prefix: 
        for obj in bucket.objects.filter(Prefix=prefix):
            # Here we are copying the directory structure from s3 bucket.
            if not os.path.exists(os.path.join('data',os.path.dirname(obj.key))):
                os.makedirs(os.path.join('data',os.path.dirname(obj.key)))
            
            dest_path = os.path.join('data',obj.key)
            try:
                bucket.download_file(obj.key,dest_path)
            except:
                print(f"We cannot download{obj.key} from {prefix}!")

if __name__ == "__main__":
    download_data(Prefix=['train'])