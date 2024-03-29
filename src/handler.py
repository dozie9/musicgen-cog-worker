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
from firebase_admin import credentials, initialize_app, storage, firestore
from runpod.serverless.utils.rp_validator import validate

LOCAL_URL = "http://127.0.0.1:5000"

cog_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
cog_session.mount('http://', HTTPAdapter(max_retries=retries))

logger = RunPodLogger()

SERVICE_CERT = json.loads(os.environ["FIREBASE_KEY"])
SADTALKER_SERVICE_CERT = json.loads(os.environ["SADTALKER_FIREBASE_KEY"])
STORAGE_BUCKET = os.environ["STORAGE_BUCKET"]

cred_obj = credentials.Certificate(SERVICE_CERT)
sad_cred_obj = credentials.Certificate(SADTALKER_SERVICE_CERT)

default_app = initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET}, name='musicgen')
sad_app = initialize_app(sad_cred_obj, name='sadtalker')

INPUT_SCHEMA = {
    "seed": {
        "type": int,
        "title": "Seed",
        "x-order": 14,
        "description": "Seed for random number generator. If None or -1, a random seed will be used."
    },
    "top_k": {
        "type": int,
        "title": "Top K",
        "default": 250,
        "x-order": 9,
        "description": "Reduces sampling to the k most likely tokens."
    },
    "top_p": {
        "type": float,
        "title": "Top P",
        "default": 0,
        "x-order": 10,
        "description": "Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used."
    },
    "prompt": {
        "type": str,
        "title": "Prompt",
        "x-order": 1,
        "description": "A description of the music you want to generate.",
        "required": True
    },
    "duration": {
        "type": int,
        "title": "Duration",
        "default": 8,
        "x-order": 3,
        "description": "Duration of the generated audio in seconds."
    },
    "input_audio": {
        "type": str,
        "title": "Input Audio",
        "format": "uri",
        "x-order": 2,
        "description": "An audio file that will influence the generated music. If `continuation` is `True`, the generated music will be a continuation of the audio file. Otherwise, the generated music will mimic the audio file's melody."
    },
    "temperature": {
        "type": float,
        "title": "Temperature",
        "default": 1,
        "x-order": 11,
        "description": "Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity."
    },
    "continuation": {
        "type": "boolean",
        "title": "Continuation",
        "default": False,
        "x-order": 4,
        "description": "If `True`, generated music will continue from `input_audio`. Otherwise, generated music will mimic `input_audio`'s melody."
    },
    "model_version": {
        "type": str,
        "title": "model_version",
        "description": "Model to use for generation",
        "default": "stereo-melody-large",
        "x-order": 0,
        "constraints": lambda model_version: model_version in [
            "stereo-melody-large",
            "stereo-large",
            "melody-large",
            "large"
        ],
    },
    "output_format": {
        "type": str,
        "title": "output_format",
        "description": "Output format for generated audio.",
        "default": "mp3",
        "x-order": 13,
        "constraints": lambda output_format: output_format in [
            "wav",
            "mp3"
        ],
    },
    "continuation_end": {
        "type": int,
        "title": "Continuation End",
        "minimum": 0,
        "x-order": 6,
        "description": "End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip."
    },
    "continuation_start": {
        "type": int,
        "title": "Continuation Start",
        "default": 0,
        "minimum": 0,
        "x-order": 5,
        "description": "Start time of the audio file to use for continuation."
    },
    "multi_band_diffusion": {
        "type": "boolean",
        "title": "Multi Band Diffusion",
        "default": False,
        "x-order": 7,
        "description": "If `True`, the EnCodec tokens will be decoded with MultiBand Diffusion. Only works with non-stereo models."
    },
    "normalization_strategy": {

        "type": str,
        "title": "normalization_strategy",
        "description": "Strategy for normalizing audio.",
        "default": "loudness",
        "x-order": 8,
        "constraints": lambda normalization_strategy: normalization_strategy in [
            "loudness",
            "clip",
            "peak",
            "rms"
        ],
    },
    "classifier_free_guidance": {
        "type": int,
        "title": "Classifier Free Guidance",
        "default": 3,
        "x-order": 12,
        "description": "Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs."
    },
    "user_id": {
        "type": str,
        "required": True
    },
}

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

    filename = f'{uuid.uuid4()}{get_extension_from_mime(mime_type)}'
    decoded_data = base64.b64decode(base64_data)

    with open(filename, 'wb') as f:
        f.write(decoded_data)

    return upload_audio(filename)


def to_firestore(audio_url, user_id):
    db = firestore.client(app=sad_app)
    push_data = {
        "uploaderId": user_id,
        # "videoCaption": prompt,
        "audioUrl": audio_url,
    }

    collection_path = "audioList"

    print("*************Starting firestore data push***************")
    update_time, firestore_push_id = db.collection(collection_path).add(
        push_data
    )

    print(update_time, firestore_push_id)


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
    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        logger.error('Error in input...')
        return {
            'errors': validated_input['errors']
        }

    json = run_inference({"input": validated_input})

    file_url = to_file(json["output"])

    to_firestore(file_url, validated_input['user_id'])

    return {
        'url': file_url,
        'user_id': validated_input['user_id']
    }


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/health-check')

    print("Cog API Service is ready. Starting RunPod serverless handler...")

    runpod.serverless.start({"handler": handler})
