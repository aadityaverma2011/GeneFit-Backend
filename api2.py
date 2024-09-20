from flask import Flask, request, jsonify
import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Set the API key directly
genai.configure(api_key="AIzaSyBckhZ-2Q3lJlLrmsDwDhefvBZ22rrvVH0")

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the pre-trained model for genetic sequence predictions
genetic_model = load_model('genetic_sequence_predictor.keras')

# Load the saved encoders
clinical_encoder = joblib.load('clinical_encoder.pkl')
disease_encoder = joblib.load('disease_encoder.pkl')

@app.route('/generate_recommendations', methods=['POST'])
def generate_recommendations():
    data = request.json
    
    # Extract parameters from the request data
    body_weight = data.get('body_weight', 0)
    height = data.get('height', 0)
    age = data.get('age', 0)
    gender = data.get('gender', '')
    diagnosed_ailment = data.get('diagnosed_ailment', 'null')
    activity_level = data.get('activity_level', 'basic')

    # Calculate BMI
    bmi = body_weight / ((height / 100) ** 2)
    
    # Create a prompt based on user data
    # Create a prompt based on user data
    prompt = f"""
    The user has the following characteristics:
    - Body weight: {body_weight} kg
    - Height: {height} centimeters
    - Age: {age} years
    - Gender: {gender}
    - Diagnosed Disease: {diagnosed_ailment}
    - Activity level: {activity_level} (out of basic(being zero to little activity),moderate(normal exercising), high(athlete levels))
    - BMI: {bmi:.2f}
    Do not incorporate any kind of text decoration in your response
    Be brutally honest and Provide 5 lifestyle recommendations based on the above details, including and in the format given :
    1. Health : Response in 3 words based on overweight,fit,underweight,obese categorisation.
    2. Recommended Calories Intake : Response in "int"-int range
    3. Recommended protein intake: Response g.
    4. Other macronutrient: Carbs- Response g, Fats: Response g, Fibre: Response g.
    5. General Tips for treating the diagnosed disease through lifestyle changes whilst keeping the current health parameters in mind. Also make sure this is a descriptive response does not exceed 40 words. Format : Diagnosed Disease : Disease_name, Response in no more than 40 words

    Dont give any additional title to the response text.
    """

    # Call the model to generate the response
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,        # Only 1 response
            max_output_tokens=500,    # Limit the response to 500 tokens
            temperature=0.7,          # Moderate creativity
        ),
    )

    # Return the response text as JSON
    return jsonify({'response': response.text.strip()})


@app.route('/predict_from_csv', methods=['POST'])
def predict_from_csv():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if the file is a CSV
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    # Load the CSV file into a DataFrame
    test_df = pd.read_csv(file)

    # Preprocessing the test data
    test_df['chromosome'].fillna('Unknown', inplace=True)
    test_df['position'].fillna(-1, inplace=True)
    test_df['ref_allele'].fillna('Unknown', inplace=True)
    test_df['alt_allele'].fillna('Unknown', inplace=True)

    # Frequency Encoding
    def frequency_encoding(df, column):
        freq = df[column].value_counts()
        df[column + '_freq'] = df[column].map(freq)
        return df

    test_df = frequency_encoding(test_df, 'chromosome')
    test_df = frequency_encoding(test_df, 'ref_allele')
    test_df = frequency_encoding(test_df, 'alt_allele')

    # Select features
    X_test = test_df[['chromosome_freq', 'position', 'ref_allele_freq', 'alt_allele_freq']]

    # Convert to numpy arrays
    X_test = np.asarray(X_test).astype('float32')

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=4)
    X_test = pca.fit_transform(X_test)

    # Make predictions on the test set
    predictions = genetic_model.predict(X_test)

    # Decode predictions
    def decode_predictions(clinical_preds, disease_preds):
        clinical_labels = clinical_encoder.inverse_transform(np.argmax(clinical_preds, axis=1))
        disease_labels = disease_encoder.inverse_transform(np.argmax(disease_preds, axis=1))
        return clinical_labels, disease_labels

    # Decode the predictions
    clinical_pred_labels, disease_pred_labels = decode_predictions(predictions[0], predictions[1])

    # Prepare the response with the first 5 predictions
    results = []
    for i in range(min(5, len(clinical_pred_labels))):
        results.append({
            'Clinical_Significance_Prediction': clinical_pred_labels[i],
            'Disease_Prediction': disease_pred_labels[i]
        })

    return jsonify(results)


# New endpoint: /combined_analysis
@app.route('/combined_analysis', methods=['POST'])
def combined_analysis():
    data = request.json
    file = request.files.get('file')

    # Extract user data
    body_weight = data.get('body_weight', 0)
    height = data.get('height', 0)
    age = data.get('age', 0)
    gender = data.get('gender', '')
    activity_level = data.get('activity_level', 'basic')

    # Check if file was uploaded
    if not file or not file.filename.endswith('.csv'):
        return jsonify({'error': 'CSV file required'}), 400

    # Load and process the CSV file as in /predict_from_csv
    test_df = pd.read_csv(file)
    test_df['chromosome'].fillna('Unknown', inplace=True)
    test_df['position'].fillna(-1, inplace=True)
    test_df['ref_allele'].fillna('Unknown', inplace=True)
    test_df['alt_allele'].fillna('Unknown', inplace=True)
    test_df = frequency_encoding(test_df, 'chromosome')
    test_df = frequency_encoding(test_df, 'ref_allele')
    test_df = frequency_encoding(test_df, 'alt_allele')

    X_test = test_df[['chromosome_freq', 'position', 'ref_allele_freq', 'alt_allele_freq']]
    X_test = np.asarray(X_test).astype('float32')

    # PCA and predictions
    pca = PCA(n_components=4)
    X_test = pca.fit_transform(X_test)
    predictions = genetic_model.predict(X_test)
    clinical_pred_labels, disease_pred_labels = decode_predictions(predictions[0], predictions[1])

    # Get the first 5 predictions
    combined_results = []
    for i in range(min(5, len(clinical_pred_labels))):
        combined_results.append({
            'clinical_significance': clinical_pred_labels[i],
            'disease': disease_pred_labels[i]
        })

    # Combine predictions and user details into the AI prompt
    prompt = f"""
    The user has the following characteristics:
    - Body weight: {body_weight} kg
    - Height: {height} cm
    - Age: {age} years
    - Gender: {gender}
    - Activity level: {activity_level}

    Based on these characteristics and the following disease predictions, generate a short lifestyle advice (30-40 words) for each potential disease occurrence:
    1. Clinical Significance: {combined_results[0]['clinical_significance']}, Disease: {combined_results[0]['disease']}
    2. Clinical Significance: {combined_results[1]['clinical_significance']}, Disease: {combined_results[1]['disease']}
    3. Clinical Significance: {combined_results[2]['clinical_significance']}, Disease: {combined_results[2]['disease']}
    4. Clinical Significance: {combined_results[3]['clinical_significance']}, Disease: {combined_results[3]['disease']}
    5. Clinical Significance: {combined_results[4]['clinical_significance']}, Disease: {combined_results[4]['disease']}
    """

    # Generate AI response
    ai_response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=500,
            temperature=0.7,
        ),
    )

    return jsonify({'ai_response': ai_response.text.strip()})


if __name__ == '__main__':
    app.run(port=8087, debug=True)
