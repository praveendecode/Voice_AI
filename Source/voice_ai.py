
from transformers import pipeline

import warnings

def get_transcription(filename: str):

    warnings.filterwarnings('ignore')

    if type(filename) == str:

        pipe = pipeline("automatic-speech-recognition", model="Praveendecode/finetuned-whishper-small-marathi") # Load pre-trained model

        result = pipe(filename, generate_kwargs={"language": "marathi"}) # Get transcription result

        transcription = result['text']

        return transcription

    else:
        raise ValueError('Provide a valid filename')
