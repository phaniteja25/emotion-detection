
import streamlit as st
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

# Paths to model and tokenizer
model_p = r"./finetuned_model"
token_p = r"./finetuned_model"

# Load the model and tokenizer
@st.cache_resource
def load_model():
    try:
        model = XLMRobertaForSequenceClassification.from_pretrained(model_p)
        tokenizer = XLMRobertaTokenizer.from_pretrained(token_p)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {e}")
        return None, None

# Initialize translation service
# translator = GoogleTranslator(source='auto', target='en')

# Define the label mapping
label_mapping = {
    0: ("Anger", "😠"),
    1: ("Fear", "😨"),
    2: ("Joy", "😊"),
    3: ("Sadness", "😢"),
    4: ("Surprise", "😲")
}

# Main app layout
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide",
)

# Sidebar layout
with st.sidebar:
    st.title("Emotion Detection App")
    st.write("This app predicts the emotion from the given text. It supports multilingual inputs and detects emotions like joy, sadness, anger, fear, and surprise.")
    st.write("### How it works:")
    st.write("1. Enter your text in any language.")
    st.write("2. The app will translate it to English.")
    st.write("3. The model predicts the emotion and displays it with an emoji.")
    st.write("---")
    st.markdown("### Powered by:")
    st.write("- **Transformers Library**")
    st.write("- **Deep Translator**")
    st.write("---")

# Main section
st.title("🌟 Emotion Detection")
st.subheader("Predict emotions with multilingual text input")
st.write("Enter a sentence below and let the model analyze its emotion!")

# Load model and tokenizer
model, tokenizer = load_model()
if not model or not tokenizer:
    st.stop()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# User input section
col1, col2 = st.columns([3, 2])

with col1:
    user_input = st.text_area(
        "Type a sentence here:",
        placeholder="Enter text in English, Hindi, Spanish, or any language...",
        height=150,
    )

with col2:
    st.write("### Example inputs:")
    st.markdown("- **English**: I am so happy today!")
    st.markdown("- **Hindi**: आज का दिन बहुत अच्छा है।")
    st.markdown("- **Spanish**: Estoy muy emocionado por mañana.")

# Predict emotion
if user_input:
    try:
        # Translation
        # user_input_translated = translator.translate(user_input)
        
        # Display translation
        # st.write("### Translated Text:")
        # st.success(user_input_translated)

        # Tokenize and predict
        inputs = tokenizer(
            user_input, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to device

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
            predicted_emotion = label_mapping[predicted_label]

        # Display the result
        st.write("### Predicted Emotion:")
        st.markdown(f"<h1 style='text-align: center; color: green;'>{predicted_emotion[1]} {predicted_emotion[0]}</h1>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter text to analyze emotion.")

# Footer
st.markdown(
    """
    ---
    **Note**: This model is a proof-of-concept and may not capture all nuances of emotion in complex sentences.
    """
)

