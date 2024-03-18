import base64
import mimetypes
import time
import subprocess
import json
import os
import uuid

import runpod
from runpod.serverless.modules.rp_logger import RunPodLogger
import requests
from requests.adapters import HTTPAdapter, Retry
from firebase_admin import credentials, initialize_app, storage

LOCAL_URL = "http://127.0.0.1:5000"

cog_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
cog_session.mount('http://', HTTPAdapter(max_retries=retries))

logger = RunPodLogger()

SERVICE_CERT = json.loads(os.environ["FIREBASE_KEY"])
STORAGE_BUCKET = os.environ["STORAGE_BUCKET"]
cred_obj = credentials.Certificate(SERVICE_CERT)
initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET})

# ----------------------------- Start API Service ---------------------------- #
# Call "python -m cog.server.http" in a subprocess to start the API service.
subprocess.Popen(["python", "-m", "cog.server.http"])


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            health = requests.get(url, timeout=120)
            status = health.json()["status"]

            if status == "READY":
                time.sleep(1)
                return

        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    response = cog_session.post(url=f'{LOCAL_URL}/predictions',
                                json=inference_request, timeout=600)
    return response.json()


def get_extension_from_mime(mime_type):
    extension = mimetypes.guess_extension(mime_type)
    return extension


def to_file(data: str):
    # bs4_code = data.split(';base64,')[-1]

    # Splitting the input string to get the MIME type and the base64 data
    split_data = data.split(",")
    mime_type = split_data[0].split(":")[1].split(';')[0]
    base64_data = split_data[1]

    filename = f'voice_{uuid.uuid4()}{get_extension_from_mime(mime_type)}'
    decoded_data = base64.b64decode(base64_data)

    with open(filename, 'wb') as f:
        f.write(decoded_data)

    return upload_audio(filename)


def upload_audio(filename):
    destination_blob_name = f'musicgen/{filename}'
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(filename)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    logger.info("File uploaded to firebase...")
    return blob.public_url



# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    json = run_inference({"input": event["input"]})

    file_url = to_file(json["output"])

    return file_url


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/health-check')

    print("Cog API Service is ready. Starting RunPod serverless handler...")

    runpod.serverless.start({"handler": handler})
