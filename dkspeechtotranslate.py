#!/usr/bin/env python
# coding: utf-8

# Before running this script, ensure the following prerequisites are met:

# 1) You need to download and run OLLama.ai or another tool to host Large Language Models locally. Find it at: https://ollama.ai
#    Ensure that it's set up and running on your local machine and accessible.

# 2) Download the Whisper library from OpenAI. You can find it at: https://github.com/openai/whisper
#    Follow the installation instructions provided in the Whisper repository to set it up.


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

# Function to transcribe a video using the Whisper model
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

# Main function to process the provided filenames
def main(filenames):
    for file in filenames:
        transcription_text = transcribe_video(file)
        if transcription_text is not None:
            generate('translate this english text into spanish. : ' + transcription_text)  # Call generate here

if __name__ == "__main__":
    # This script can be executed from the command line as follows:
    # python3 dkspeechtotranslate.py {videofilename}
    # Replace {videofilename} with the path to the video file you want to transcribe and translate.
    
    # Parse the provided file paths from the command line arguments
    filenames = sys.argv[1:]
    
    # Call the 'main' function to process the provided filenames
    main(filenames)