import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detector 🌿",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for improved aesthetics
st.markdown("""
    <style>
        .stApp {
            background-color: #121212;
            max-width: 1000px;
            margin: auto;
            padding: 30px;
            border-radius: 12px;
        }
        .upload-text {
            font-size: 18px;
            color: #e0e0e0;
            margin-bottom: 20px;
        }
        .result-box {
            padding: 20px;
            background-color: #2e7d32;
            border: 2px solid #66bb6a;
            border-radius: 10px;
            color: #fff;
        }
        .low-confidence {
            padding: 20px;
            background-color: #ffecb3;
            border: 2px solid #ffeeba;
            border-radius: 10px;
            color: #333;
        }
        .error-box {
            padding: 20px;
            background-color: #f8d7da;
            border: 2px solid #f5c6cb;
            border-radius: 10px;
            color: #721c24;
        }
        .stButton>button {
            background-color: #66bb6a;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #4caf50;
        }
        .stFileUploader>label {
            background-color: #4caf50;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .stFileUploader>label:hover {
            background-color: #388e3c;
        }
    </style>
""", unsafe_allow_html=True)

# Define paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "cnn_plant_disease_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load model
if not os.path.exists(model_path):
    st.error(f"Model not found at: {model_path}")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load class indices
try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Error loading class indices: {str(e)}")
    st.stop()

def is_valid_plant_image(image):
    try:
        img_array = np.array(image)
        avg_brightness = np.mean(img_array)

        if avg_brightness < 20 or avg_brightness > 245:
            return False, "The image is too dark or too bright."

        width, height = image.size
        if width < 100 or height < 100:
            return False, "The image is too small. Please upload a larger plant leaf image."

        if len(img_array.shape) == 3:
            hsv = tf.image.rgb_to_hsv(img_array / 255.0)
            green_mask = tf.reduce_sum(tf.cast(
                (hsv[..., 0] >= 0.15) &
                (hsv[..., 0] <= 0.5) &
                (hsv[..., 1] >= 0.1) &
                (hsv[..., 2] >= 0.2),
                tf.float32
            ))
            total_pixels = width * height
            green_ratio = green_mask / total_pixels
            if green_ratio < 0.10:
                return False, "The uploaded image doesn't seem to contain enough green area to be a plant leaf."

        return True, "Valid image."
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype('float32') / 255.

def predict_image_class(image):
    is_valid, message = is_valid_plant_image(image)
    if not is_valid:
        return None, None, message

    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(predictions[0][predicted_class_index])

    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Class")

    # Adjust confidence threshold for unknown categories
    if predicted_class_name == "Unknown Class":
        confidence = 0.45  # Arbitrarily low confidence for unknown categories
        return predicted_class_name, confidence, "Low confidence — Unfamiliar class detected."

    if confidence < 0.60:
        return predicted_class_name, confidence, "Low confidence — might not match dataset class."

    return predicted_class_name, confidence, "Prediction successful."

# Disease-specific suggestions
DISEASE_SUGGESTIONS = {
    "Apple___Apple_scab": [
        "Remove and destroy fallen infected leaves",
        "Apply fungicide sprays in early spring",
        "Improve air circulation by pruning",
        "Plant scab-resistant apple varieties"
    ],
    "Apple___Black_rot": [
        "Remove mummified fruits and cankers",
        "Prune out dead or diseased branches",
        "Apply fungicide during growing season",
        "Maintain good sanitation practices"
    ],
    "Apple___Cedar_apple_rust": [
        "Remove nearby cedar trees if possible",
        "Apply preventative fungicides",
        "Plant rust-resistant apple varieties",
        "Monitor trees in spring for early signs"
    ],
    "Grape___Black_rot": [
        "Remove mummified berries and infected leaves",
        "Improve air circulation in the vineyard",
        "Apply fungicide before and after bloom",
        "Maintain proper canopy management"
    ],
    "Tomato___Late_blight": [
        "Remove and destroy infected plants",
        "Apply copper-based fungicide",
        "Avoid overhead watering",
        "Space plants for good air circulation"
    ],
    "Tomato___Early_blight": [
        "Remove lower infected leaves",
        "Mulch around plants",
        "Apply appropriate fungicide",
        "Rotate crops annually"
    ],
    "Potato___Late_blight": [
        "Remove infected plants and destroy them",
        "Apply fungicide preventatively",
        "Plant resistant varieties",
        "Monitor weather conditions"
    ],
    "Tomato___Leaf_Mold": [
        "Improve air circulation",
        "Reduce humidity levels",
        "Remove infected leaves",
        "Apply appropriate fungicide"
    ],
    "Tomato___Bacterial_spot": [
        "Remove infected plants",
        "Avoid overhead irrigation",
        "Use copper-based sprays",
        "Practice crop rotation"
    ]
}

HEALTHY_SUGGESTIONS = [
    "Continue regular monitoring.",
    "Maintain balanced watering.",
    "Ensure proper light and soil conditions.",
    "Preventive care with timely fertilizers."
]

DEFAULT_SUGGESTIONS = [
    "Isolate the plant from others.",
    "Trim and remove infected leaves or branches.",
    "Apply general-purpose antifungal or pesticide sprays.",
    "Consult a local agricultural expert."
]

def get_suggestions(prediction):
    if "healthy" in prediction.lower():
        return "Healthy Plant Care:", HEALTHY_SUGGESTIONS
    elif prediction in DISEASE_SUGGESTIONS:
        return "Treatment Suggestions:", DISEASE_SUGGESTIONS[prediction]
    else:
        return "General Suggestions:", DEFAULT_SUGGESTIONS

# Streamlit App
st.title('🌿 Plant Disease Classifier')
st.markdown("""
    <div class="upload-text">
    Upload a clear image of a plant leaf to detect diseases. For best results:
    <ul>
        <li>Use well-lit, clear images</li>
        <li>Ensure the leaf is the main focus of the image</li>
        <li>Avoid blurry or dark images</li>
        <li>Use images with natural lighting</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence_result' not in st.session_state:
    st.session_state.confidence_result = None

# Ask for the user's name
user_name = st.text_input("Enter your name:")

if user_name:
    # Reset uploaded image button
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Handle image prediction and processing
    if uploaded_image:
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img, caption="Uploaded Image")

        with col2:
            if st.button('Classify'):
                with st.spinner('Analyzing image...'):
                    # Preprocess the uploaded image and predict the class
                    prediction, confidence, message = predict_image_class(image)

                    if prediction and confidence is not None:
                        # Store results in session state
                        st.session_state.prediction_result = prediction
                        st.session_state.confidence_result = confidence
                        
                        if confidence < 0.60:
                            st.warning(f'Low confidence: {message}')
                        st.success(f'Prediction: {prediction.replace("___", " - ")}')
                        st.info(f'Confidence: {confidence:.2%}')

                        # Get and display suggestions
                        with st.expander("View Recommendations", expanded=True):
                            title, suggestions = get_suggestions(prediction)
                            st.markdown(f"### {title}")
                            for suggestion in suggestions:
                                st.markdown(f"• {suggestion}")
                    else:
                        st.error(message)
                        if "doesn't appear to be a plant leaf" in message:
                            st.warning("⚠️ Please ensure:")
                            st.markdown("""
                                - The leaf is clearly visible in the image
                                - The image is well-lit
                                - The leaf has some green coloring
                                - The image is not too dark or overexposed
                                - The leaf fills a good portion of the image
                            """)
                            st.info("💡 Try using an image where the leaf is the main focus and is photographed against a plain background.")

            # Option to reset the uploaded image
            if st.button('Reset Image'):
                uploaded_image = None
                st.session_state.prediction_result = None
                st.session_state.confidence_result = None
                st.experimental_rerun()

            # Save Report Button (only show if we have valid predictions)
            if st.session_state.prediction_result and st.session_state.confidence_result:
                if st.button('Save Report'):
                    report = f"User: {user_name}\n\nPrediction: {st.session_state.prediction_result}\nConfidence: {st.session_state.confidence_result:.2%}\n\nSuggestions:\n"
                    suggestions_title, suggestions_list = get_suggestions(st.session_state.prediction_result)
                    for suggestion in suggestions_list:
                        report += f"- {suggestion}\n"

                    report_filename = f"{user_name}_plant_disease_report.txt"
                    with open(report_filename, 'w') as f:
                        f.write(report)

                    st.success(f"Report saved successfully as {report_filename}")
else:
    st.info("Please enter your name to get started.")
