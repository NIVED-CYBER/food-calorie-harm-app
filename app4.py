# Install NLTK data
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Import libraries
import streamlit as st
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import spacy
from nltk.corpus import wordnet
import pytesseract
import io
import torch

# Set page config
st.set_page_config(
    page_title="Food Analysis App",
    page_icon="🍽️",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = False

# Load models
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    MODEL_NAME = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    return nlp, processor, model

nlp, processor, model = load_models()

# Constants
USDA_API_KEY = "b7pt6Io9sPCr9DJutuydgH8tuocaUHQzfwZeHJDf"
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
OPENFOODFACTS_API = "https://world.openfoodfacts.org/api/v0/product/"

FOOD_LIST = {
    "apple", "banana", "bread", "chocolate", "salad", "tomato", "cucumber",
    "rice", "chicken", "fish", "pizza", "burger", "pasta", "cheese", "egg",
    "carrots", "pork", "beans", "cake", "sandwich", "strawberry", "yogurt",
    "mushroom", "lettuce", "onion", "beef", "milk", "orange", "avocado",
}

# Food Calorie Functions
def is_food_word(word):
    word = word.lower()
    if word in FOOD_LIST:
        return True
    
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    for syn in synsets:
        if "food" in syn.lexname() or "edible" in syn.definition():
            return True
    return False

def fetch_nutrition(food_name):
    params = {
        "query": food_name,
        "api_key": USDA_API_KEY,
        "pageSize": 1
    }
    
    response = requests.get(USDA_API_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "foods" in data and data["foods"]:
            food_data = data["foods"][0]["foodNutrients"]
            nutrients = {nutrient["nutrientName"]: nutrient["value"] for nutrient in food_data}
            
            return {
                "calories": nutrients.get("Energy", 0),
                "protein": nutrients.get("Protein", 0),
                "fat": nutrients.get("Total lipid (fat)", 0),
                "carbs": nutrients.get("Carbohydrate, by difference", 0)
            }
    return None

def generate_caption(image):
    try:
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error processing image: {e}"

def extract_food_items(description):
    doc = nlp(description.lower())
    detected_foods = {word.text for word in doc if is_food_word(word.text)}
    return list(detected_foods)

# Harmful Ingredients Functions
def extract_text_from_image(image):
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error with Tesseract OCR: {str(e)}")
        return ""

def check_ingredients(ingredients):
    harmful_ingredients = []
    ingredients_list = [ingredient.strip().lower() for ingredient in ingredients.split(',')]
    
    for ingredient in ingredients_list:
        if not ingredient:  # Skip empty ingredients
            continue
            
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={ingredient}&search_simple=1&action=process&json=1"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'products' in data and data['products']:
                product = data['products'][0]
                additives = product.get('additives_tags', [])
                
                if additives:
                    harmful_ingredients.append((ingredient, additives))
        except Exception as e:
            st.error(f"Error processing ingredient {ingredient}: {str(e)}")
    
    return harmful_ingredients

# Streamlit UI
st.title("🍽️ NUTRIFO")
st.write("Upload a food image to analyze its calories and check for harmful ingredients.")

# Create tabs
tab1, tab2 = st.tabs(["Calorie Estimation", "Ingredients Analysis"])

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Calorie Estimation Tab
    with tab1:
        st.header("Calorie Estimation")
        
        # Generate caption and extract food items
        with st.spinner("Analyzing image..."):
            caption = generate_caption(image)
            food_items = extract_food_items(caption)
        
        st.write(f"**Generated Caption:** {caption}")
        
        if food_items:
            st.write("**Detected Food Items:**", ", ".join(food_items))
            
            # Create input fields for weights
            st.write("### Enter Food Weights")
            weights = {}
            
            col1, col2 = st.columns(2)
            for i, food in enumerate(food_items):
                with col1 if i % 2 == 0 else col2:
                    weights[food] = st.number_input(
                        f"Enter weight for {food} (grams):",
                        min_value=0.0,
                        value=100.0,
                        step=10.0,
                        key=food
                    )
            
            if st.button("Calculate Nutrition", key="calc_nutrition"):
                with st.spinner("Fetching nutritional information..."):
                    st.write("### Nutritional Information")
                    
                    total_calories = 0
                    nutrition_data = {}
                    
                    # Calculate nutrition for each food item
                    for food in food_items:
                        data = fetch_nutrition(food)
                        if data:
                            weight = weights[food]
                            calories = (data["calories"] * weight) / 100
                            nutrition_data[food] = {
                                "weight": weight,
                                "calories": calories,
                                "protein": (data["protein"] * weight) / 100,
                                "fat": (data["fat"] * weight) / 100,
                                "carbs": (data["carbs"] * weight) / 100
                            }
                            total_calories += calories
                    
                    # Display results in a clean format
                    col1, col2 = st.columns(2)
                    with col1:
                        for food, details in nutrition_data.items():
                            st.write(f"**{food.capitalize()}** ({details['weight']}g):")
                            st.write(f"- Calories: {details['calories']:.1f} kcal")
                            st.write(f"- Protein: {details['protein']:.1f}g")
                            st.write(f"- Fat: {details['fat']:.1f}g")
                            st.write(f"- Carbs: {details['carbs']:.1f}g")
                            st.write("---")
                    
                    with col2:
                        st.metric("Total Calories", f"{total_calories:.1f} kcal")
        else:
            st.warning("No recognizable food items found in the image.")
    
    # Ingredients Analysis Tab
    with tab2:
        st.header("Ingredients Analysis")
        
        # Extract text from image
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(image)
        
        # Show extracted text and allow editing
        ingredients_text = st.text_area(
            "Extracted Ingredients (edit if needed):",
            value=extracted_text,
            height=150
        )
        
        if st.button("Analyze Ingredients", key="analyze_ingredients"):
            if ingredients_text.strip():
                with st.spinner("Analyzing ingredients..."):
                    harmful_found = check_ingredients(ingredients_text)
                    
                    if harmful_found:
                        st.error("⚠️ Harmful Ingredients Detected:")
                        for item, additives in harmful_found:
                            with st.expander(f"{item.capitalize()}"):
                                st.write("Found additives:")
                                for additive in additives:
                                    st.write(f"- {additive}")
                    else:
                        st.success("✅ No harmful ingredients detected.")
            else:
                st.warning("No ingredients text found or provided.")

else:
    st.info("Please upload an image to begin analysis.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
    ### How to use:
    1. Upload a food image using the sidebar
    2. Switch between tabs to:
        - Calculate calories and nutrition
        - Check for harmful ingredients
    3. Follow the instructions in each tab
    
    ### Note:
    - Image analysis may take a few moments
    - You can edit extracted ingredients text
    - Nutritional values are estimates
""")
