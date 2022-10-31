# Importing packages
import boto3
import os

def download_data(prefix_list:list[str]) -> None:
    ''' Here we are downloading our dataset from the given bucket 

        args: 
            Prefix(list[str]): subdirectories inside of our main bucket in s3.
            may want to include functionality for 
    ''' 
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('mle9-spotify-wakeword-data')
    
    for prefix in prefix_list: 
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
    download_data(prefix_list=['test'])