from util import get_json_file_list, upload_to_s3, read_json_file
import pprint   
import os
from ollama import Client

def create_messages():
    # Get responses
    file_list = get_json_file_list(os.getenv("MESSAGES_BUCKET_NAME"))

    # Batches of 10k, upload to s3
    user_messages = []
    batch_number = 0
    batch_size = 1000
    for file_name in file_list:
        response = read_json_file(os.getenv("MESSAGES_BUCKET_NAME"), file_name)
        messages = response["messages"]
        for message in messages:
            if message["sender_name"] == os.getenv("USER_NAME"):
                if "content" in message:
                    user_messages.append(message["content"])

                if len(user_messages) == batch_size:
                    upload_to_s3(os.getenv("USER_MESSAGES_BUCKET"), f"user_messages_{batch_number}.json", {"data": user_messages})
                    batch_number += 1
                    user_messages = []

    print(f"Uploading {len(user_messages)} messages to s3")
    if len(user_messages) > 0:
        upload_to_s3(os.getenv("USER_MESSAGES_BUCKET"), f"user_messages_{batch_number}.json", {"data": user_messages})

    
if __name__ == "__main__":
    create_messages()