#!/usr/bin/env python
# coding: utf-8

import sys
import whisper
import json
import requests

def generate(prompt):
    llmmodel = 'llama2'
    try:
        r = requests.post('http://localhost:11434/api/generate',
                          json={
                              'model': llmmodel,
                              'prompt': prompt,
                          },
                          stream=True)
        r.raise_for_status()
        response_text = str(r.text)
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")

    print(llmmodel + " translated text :")
    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        # the response streams one token at a time, print that as we receive it
        print(response_part, end='', flush=True)

        if 'error' in body:
            raise Exception(body['error'])


def transcribe_video(filename):
    # Ensure you have setup the whisper model from the official repository:
    # https://github.com/openai/whisper
    
    try:   
        # Load the Whisper model
        modelname = "medium"
        model = whisper.load_model(modelname)
        result = model.transcribe(filename, fp16=False, language="en")
        transcription_text = result['text']
        print("Whisper model:" + modelname + " transcribed text :")
        print(transcription_text)
              
        return transcription_text 
        
    except Exception as e:
        print(f"Error during transcription: {e}")

def main(filenames):
    for file in filenames:
        transcription_text = transcribe_video(file)
        if transcription_text is not None:
            generate('translate this english text into spanish. : ' + transcription_text)  # Call generate here

if __name__ == "__main__":
    # Assuming file paths are passed as separate command-line arguments
    filenames = sys.argv[1:]
    main(filenames)