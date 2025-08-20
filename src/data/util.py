import boto3
import json
import sys
import os

s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                  aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))


# Upload data to S3
def upload_to_s3(bucket_name, file_name, data):
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=json.dumps(data))


# List JSON files
def get_json_file_list(bucket_name):
    file_names = []
    paginator = s3.get_paginator('list_objects_v2')
    
    try:
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    if file_key.lower().endswith('.json'):
                        file_names.append(file_key)
        
        if not file_names:
            print("No JSON files found in the bucket.")
            return []
        
        file_names.sort()
        print(f"Found {len(file_names)} JSON files in bucket '{bucket_name}'")
        
        return file_names
        
    except Exception as e:
        print(f"Error reading from S3 bucket '{bucket_name}': {str(e)}")
        return []

# Read JSON file
def read_json_file(bucket_name, file_name):
    try:
        file_obj = s3.get_object(Bucket=bucket_name, Key=file_name)
        
        file_content = file_obj['Body'].read().decode('utf-8')
        parsed_json = json.loads(file_content)
        
        return parsed_json
        
    except Exception as e:
        print(f"Error reading file '{file_name}': {str(e)}")
        return None

# Check non-ASCII characters
def has_non_ascii(s):
    return any(ord(c) > 127 for c in s)
