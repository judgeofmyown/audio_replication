import csv
from flask import Flask, request, jsonify
import joblib
import numpy as np
import librosa
from scipy.stats import skew, kurtosis, mode
import os

model = joblib.load('model.pkl')
DATASET_PATH = './archive/voice.csv'

def extract_audio_features(audio_path):
    print("..loading Librosa")
    y, sr = librosa.load(audio_path, sr=None) 
    
    # Frequency domain features
    print("frequecy domain features")
    stft = np.abs(librosa.stft(y))
    print(f"len stft: {len(stft)}")
    freqs = librosa.fft_frequencies(sr=sr)
    print(f"freqs: {freqs}")
    S_db = librosa.amplitude_to_db(stft, ref=np.max)
    print(f"s_db: {S_db}")

    # Spectral Features
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    print(f"centroid: {centroid}")
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    print(f"flatness: {flatness}")
    entropy = -np.sum(S_db * np.log2(S_db + 1e-10), axis=0) / np.log2(S_db.shape[0])
    print(f"entropy: {entropy}")

    # Fundamental frequency
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    print(f"pitches: {pitches}")
    print(f"magnitudes: {magnitudes}")
    fundamental_freqs = pitches[pitches > 0]
    print(f"fundamental freq: {fundamental_freqs}")
    meanfun = np.mean(fundamental_freqs)
    print(f"meanfun", meanfun)
    minfun = np.min(fundamental_freqs)
    print(f"minfun: {minfun}")
    maxfun = np.max(fundamental_freqs)
    print(f"maxfun: {maxfun}")

    # Dominant frequency
    dominant_freqs = freqs[np.argmax(magnitudes, axis=0)]
    print(f"dominant freq: {dominant_freqs}")
    dominant_freqs = dominant_freqs[dominant_freqs > 0]
    print(f"dominant freqs: {dominant_freqs}")
    meandom = np.mean(dominant_freqs)
    print(f"meandom: {meandom}")
    mindom = np.min(dominant_freqs)
    print(f"mindom: {mindom}")
    maxdom = np.max(dominant_freqs)
    print(f"maxdom: {maxdom}")
    dfrange = maxdom - mindom
    print(f"dfrange: {dfrange}")

    # Modulation Index
    modindx = np.sum(np.abs(np.diff(fundamental_freqs))) / (maxfun - minfun)
    print(f"modindx: {modindx}")

    # Statistical Features
    meanfreq = np.mean(freqs)
    print(f"meanfreq: {meanfreq}")
    sdfreq = np.std(freqs)
    print(f"sd: {sdfreq}")
    medianfreq = np.median(freqs)
    print(f"medianfreq: {medianfreq}")
    q25freq = np.percentile(freqs, 25)
    print(f"q25freq: {q25freq}")
    q75freq = np.percentile(freqs, 75)
    print(f"q75freq: {q75freq}")
    iqr = q75freq - q25freq
    print(f"IQR: {iqr}")
    skewness = skew(freqs)
    print(f"skew: {skewness}")
    kurt = kurtosis(freqs)
    print(f"kurt: {kurt}")
    modefreq = mode(freqs, keepdims=False).mode
    print(f"mode: {modefreq}")

    # peakf = freqs[np.argmax(S_db)]
    # print(f"peakf: {peakf}")
    features = {
        'meanfreq': meanfreq / 1000,  # Convert to kHz
        'sd': sdfreq,
        'median': medianfreq / 1000,
        'Q25': q25freq / 1000,
        'Q75': q75freq / 1000,
        'IQR': iqr / 1000,
        'skew': skewness,
        'kurt': kurt,
        'sp.ent': np.mean(entropy),
        'sfm': np.mean(flatness),
        'mode': modefreq / 1000,
        'centroid': np.mean(centroid) / 1000,
        # 'peakf': peakf / 1000,
        'meanfun': meanfun / 1000,
        'minfun': minfun / 1000,
        'maxfun': maxfun / 1000,
        'meandom': meandom / 1000,
        'mindom': mindom / 1000,
        'maxdom': maxdom / 1000,
        'dfrange': dfrange / 1000,
        'modindx': modindx
    }

    return features

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_gender():
    print(f"Predicting Gender")
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    print(f"Requesting Audio file")
    audio_file = request.files['audio']
    print(f"audio file name: {audio_file}")
    file_path = f"./uploads/{audio_file.filename}"
    print(f"File path name: {file_path}")
    print("Saving audio file...")
    audio_file.save(file_path)

    print("Extracting audio features...")
    try:
        features = extract_audio_features(file_path)
        print("Audio features extracted successfully")
        print(f"Features: {features}")
        feature_values = list(features.values())
        print("Feature values:", feature_values)
        
        prediction = model.predict(np.array(feature_values).reshape(1, -1))
        print(f"prediction: {prediction}")
        gender = "male" if prediction[0] == 0 else "female"
        print(f"Gender: {gender}")
        label = 0 if gender == "male" else 1

        print("Appending to CSV...")
        append_to_csv(feature_values, label)

        print("CSV appending completed successfully")
        return jsonify({'gender': gender})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def append_to_csv(features, label):
    file_exists = os.path.isfile(DATASET_PATH)
    print(f"file exist: {file_exists}")

    with open(DATASET_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            header = [
                'meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 
                'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 
                'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'
            ]
            writer.writerow(header)
        
        writer.writerow(features + [label])

if __name__ == '__main__':
    app.run(debug=True)
