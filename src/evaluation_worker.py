import boto3
import json
import time
import os, sys
import warnings
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import db, Submission
from evaluation import Model_Evaluator
import gc
gc.set_debug(gc.DEBUG_LEAK)


try:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('advex')
    s3_client = boto3.client('s3')
    sqs = boto3.client('sqs')
    resp = sqs.get_queue_url(QueueName='advex')
    queue_url = resp['QueueUrl']
except:
    # should raise error here
    warnings.warn("sqs not started", UserWarning)



def update_feedback(submission_id, feedback=None, status=None):
    print('Writing feedback.')
    submission = Submission.query.get(submission_id)
    if feedback is not None:
        submission.feedback = feedback
    if status is not None:
        submission.status = status
    db.session.commit()


def evaluate_job(job):
    print('Evaluating model.')

    feedback={}
    try:
        submission_id = job['submission_id']
        model_file = job['s3_model_key']
        index_file = job['s3_index_key']
    
        update_feedback(submission_id, status='Running')
        
        if not model_file.endswith('.h5'):
            raise Exception('Model file has to have .h5 as its extension.')

        if not index_file.endswith('.json'):
            raise Exception('Index file has to have .json as its extension')
        
    
        response_model = s3_client.head_object(Bucket='advex', Key=model_file)
        response_index = s3_client.head_object(Bucket='advex', Key=index_file)

        model_size=response_model['ContentLength']
        index_size=response_index['ContentLength']
    
        #Check 2: File Size Check
        if model_size > 1073741824: # 1 GiB
            raise Exception('.h5 file can not be bigger than 1GB.')
  
        if index_size > 102400:
            raise Exception('.json file can not be bigger than 100KB.')
 
        bucket.download_file(model_file, model_file)
        bucket.download_file(index_file, index_file)
        
        #The model file and index file are perfectly fine.
        model=Model_Evaluator(model_file, index_file, AE_path='./image_data_small/')
        feedback=model.evaluate()
    except Exception as exc:
        feedback['error']=exc.__str__()

    try:
        os.remove(model_file)
        os.remove(index_file)
    except Exception as e:
        # TODO: logging
        pass
    
    print(feedback)
    status = ('Failed' if 'error' in feedback else 'Finished')
    update_feedback(submission_id, feedback=feedback, status=status)


def main():
    while True:
        resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
        if 'Messages' not in resp:
            print('No messages received, sleep for 10s.')
            time.sleep(10)
            continue

        print('Message received.')
        message = resp['Messages'][0]
        receipt_handle = message['ReceiptHandle']
        job = json.loads(message['Body'])

        # Process job
        evaluate_job(job)

        # Delete message
        resp = sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)


if __name__ == '__main__':
    main()
