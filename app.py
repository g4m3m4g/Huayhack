import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dfrev.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

# Prepare data for prediction
def prepare_prediction_data(df):
    digit_cols = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
    
    # Create features from historical data
    X = []
    y = {col: [] for col in digit_cols}
    
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        features = list(current_row[digit_cols])
        
        X.append(features)
        for col in digit_cols:
            y[col].append(next_row[col])
    
    return X, y

# Train models for each digit
def train_digit_models(X, y, rs=42):
    models = {}
    digit_cols = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
    
    for col in digit_cols:
        model = RandomForestClassifier(n_estimators=100, random_state=rs)
        model.fit(X, y[col])
        models[col] = model
    
    return models

# Main Streamlit app
def main():
    # Set page config to improve layout
    st.set_page_config(page_title="Thai Lottery Digit Prediction", page_icon="🎫", layout="wide")
    
    # Apply custom CSS for styling
    st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 18px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stMetric {
            font-size: 20px;
        }
        .header {
            color: #FF6347;
            font-size: 36px;
            text-align: center;
            font-weight: bold;
        }
        .subheader {
            color: #4682B4;
            font-size: 24px;
            font-weight: bold;
        }
        .result-card {
            background-color: #f0f0f5;
            padding: 20px;
            margin-top: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .result-card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        .result-text {
            font-size: 30px;
            font-weight: bold;
            color: #2E8B57;
        }
        .result-index {
            font-size: 24px;
            font-weight: bold;
            color: #FF6347;
        }
        .prediction-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }
        .header-text {
            font-size: 36px;
            text-align: center;
            color: #4682B4;
            font-weight: bold;
        }
        .disclaimer {
            font-size: 16px;
            color: #FF6347;
            text-align: center;
            margin-top: 40px;
            font-style: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🎫 Thai Lottery Digit Prediction")
    
    # Load and prepare data
    df = load_data()
    X, y = prepare_prediction_data(df)
    
    # User input for random state
    random_state = st.number_input('Enter Random State for Model Training', min_value=0, max_value=100, value=42, step=1)
    
    # Train models
    models = train_digit_models(X, y, rs=random_state)
    
    # Prediction input
    st.header('🔮 Predict the Next Lottery Digits')
    
    # Input fields for current digits
    current_digits = []
    cols = st.columns(6)
    digit_labels = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']
    
    for i, col in enumerate(cols):
        current_digits.append(
            col.number_input(f'{digit_labels[i]} Digit', 
                              min_value=0, 
                              max_value=9, 
                              value=0, 
                              step=1, 
                              key=f'digit_{i}')
        )
    
    # Predict button
    if st.button('🔮 Predict Top 5 Next Digits'):
        # Store top 5 predictions as a list of 6-digit numbers
        top_5_predictions = []
        
        # For each digit (1st to 6th), get the top 5 most probable digits
        for i, model in enumerate(models.values()):
            # Get the probabilities for each digit (10 classes for digits 0 to 9)
            prob = model.predict_proba([current_digits])[0]
            
            # Get the top 5 indices based on highest probabilities
            top_5_indices = np.argsort(prob)[-5:][::-1]
            top_5_digits = [str(i) for i in top_5_indices]
            
            # For each of the top 5 predictions, construct the 6-digit number
            if not top_5_predictions:
                # Initialize the list with the top 5 digits from the first model
                top_5_predictions = [[digit] for digit in top_5_digits]
            else:
                # Append the digit to the existing 6-digit number list
                for j, prediction in enumerate(top_5_predictions):
                    prediction.append(top_5_digits[j])
        
        # Combine the list of predictions into 6-digit numbers
        top_5_predictions = [''.join(pred) for pred in top_5_predictions]
        
        # Display predictions
        st.header('✨ Predicted Top 5 Next Digits')
        
        # Wrap the predictions in a container for better styling
        prediction_container = st.container()
        with prediction_container:
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            for idx, prediction in enumerate(top_5_predictions, 1):
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-index">Prediction {idx}:</div>
                    <div class="result-text">{prediction}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    

    # Additional dataset info
    st.header('📊 Dataset Summary')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Total Records', len(df))
        st.metric('Date Range', f'{df["date"].min().date()} to {df["date"].max().date()}')
    
    with col2:
        st.metric('Digit Distribution', 'Random Forest Prediction')

    st.header('📰 5 Previous First Prize Numbers')
    st.write(df['firstprize'].astype(str).str.zfill(6).tail(5).reset_index(drop=True))

    # Disclaimer message
    st.markdown("""
    <div class="disclaimer">
        ⚠️ Disclaimer: This application is intended for entertainment purposes only. The predictions provided are based on historical data 
        and do not guarantee any future outcomes. This is not investment advice and does not encourage gambling.
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
