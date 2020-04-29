from asr import speech
import uuid,os
from flask import Flask, request, jsonify
import os, gc, requests
from flask_cors import CORS, cross_origin
import time,re
from scipy.io.wavfile import read

app = Flask(__name__)
CORS(app)

@app.route('/speech_to_speech/', methods=['POST'])
def speech_to_speech():
	file_1 = request.files['audio']
	if file_1.filename == '':
		return jsonify({'result': 'File Name_1 is blank'})
	path='./recorded_audio/' + str(uuid.uuid4())+'.wav'
	audio = os.path.join(path)
	file_1.save(audio)
	# this code convert into text 
	question=speech(path)
	print(question)
	if not question:
		print('hi')
		return jsonify({'message':"please say something"})

	return jsonify({'message':question})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4444,debug=True, use_reloader=False)