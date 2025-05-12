from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

app = Flask(__name__)

# Load models and scaler
model = tf.keras.models.load_model("malware_classifier_pe_only.keras")
defended_model = tf.keras.models.load_model("malware_classifier_defended.keras")
scaler = joblib.load("scaler_pe_only.save")

# PE features used
pe_features = [
    'Machine', 'NumberOfSections', 'TimeDateStamp', 'PointerToSymbolTable',
    'NumberOfSymbols', 'SizeOfOptionalHeader', 'Characteristics',
    'Magic', 'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode',
    'AddressOfEntryPoint', 'BaseOfCode', 'ImageBase', 'SectionAlignment',
    'FileAlignment', 'MajorOperatingSystemVersion', 'MinorOperatingSystemVersion',
    'MajorImageVersion', 'MinorImageVersion', 'MajorSubsystemVersion',
    'MinorSubsystemVersion', 'SizeOfImage', 'SizeOfHeaders', 'CheckSum',
    'Subsystem', 'DllCharacteristics', 'SizeOfStackReserve', 'SizeOfStackCommit',
    'SizeOfHeapReserve', 'SizeOfHeapCommit', 'LoaderFlags', 'NumberOfRvaAndSizes'
]

@app.route('/')
def home():
    return "👋 Welcome to the Malware Detection API!"

@app.route('/predict/features', methods=['POST'])
def predict_from_features():
    try:
        data = request.get_json()
        features = [data.get(f, 0) for f in pe_features]
        features = scaler.transform([features])
        prediction = model.predict(features)[0][0]
        label = "malware" if prediction >= 0.5 else "benign"
        return jsonify({"score": float(prediction), "classification": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict/pgd', methods=['GET'])
def pgd_attack_api():
    try:
        data = pd.read_csv("dataset_malwares.csv")
        features = [f for f in pe_features if f in data.columns]
        X = scaler.transform(data[features].values)
        y = data['Malware'].astype(int).values

        for i in range(len(y)):
            if y[i] == 1:
                test_sample = X[i]
                label = y[i]
                break

        adv = pgd_attack(model, test_sample, label)
        original = model.predict(test_sample[np.newaxis])[0][0]
        adversarial = model.predict(adv[np.newaxis])[0][0]

        return jsonify({
            "original_score": float(original),
            "adversarial_score": float(adversarial)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/pgd-defended', methods=['GET'])
def pgd_defended_api():
    try:
        data = pd.read_csv("dataset_malwares.csv")
        features = [f for f in pe_features if f in data.columns]
        X = scaler.transform(data[features].values)
        y = data['Malware'].astype(int).values

        for i in range(len(y)):
            if y[i] == 1:
                test_sample = X[i]
                label = y[i]
                break

        adv = pgd_attack(defended_model, test_sample, label)
        squeezed = feature_squeeze(adv)

        original = defended_model.predict(test_sample[np.newaxis])[0][0]
        defended = defended_model.predict(squeezed[np.newaxis])[0][0]

        return jsonify({
            "original_score": float(original),
            "defended_score": float(defended)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# PGD Attack
def pgd_attack(model, input_data, input_label, epsilon=0.3, alpha=0.01, num_iter=40):
    adv = tf.convert_to_tensor(input_data[np.newaxis], dtype=tf.float32)
    input_label = tf.convert_to_tensor([[input_label]], dtype=tf.float32)
    loss = tf.keras.losses.BinaryCrossentropy()

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            prediction = model(adv)
            loss_val = loss(input_label, prediction)
        gradient = tape.gradient(loss_val, adv)
        perturbation = alpha * tf.sign(gradient)
        adv = adv + perturbation
        delta = tf.clip_by_value(adv - input_data[np.newaxis], -epsilon, epsilon)
        adv = tf.clip_by_value(input_data[np.newaxis] + delta, -1.0, 1.0)

    return adv.numpy()[0]

# Feature Squeezing
def feature_squeeze(features):
    return np.round(features, 2)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
