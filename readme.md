# Thai Lottery Digit Prediction 🎫
![Image](https://github.com/user-attachments/assets/f254cce4-8562-4a9c-af49-78f0420b7916)

This project leverages machine learning to predict the next digits of the Thai Lottery based on historical data. It uses a Random Forest model to forecast the most probable numbers, providing users with insights into potential lottery outcomes.

## 🧑‍💻 How It Works

The application allows users to input the last drawn digits of the Thai lottery, and it uses historical data to predict the next set of lottery digits. The predictions are based on a Random Forest classifier model trained on previous lottery data, and it displays the top 5 most probable predictions.

## ⚙️ Features

- **Predict the next lottery digits**: Input the last drawn digits to predict the next ones.
- **Top 5 predictions**: Displays the top 5 most probable 6-digit numbers.
- **Dataset Summary**: Shows the dataset's date range and total records.
- **Recent Lottery Data**: Displays the last 5 first-prize numbers.
- **Disclaimer**: The predictions are based on historical data and are not guaranteed.

## 📈 Dataset

This project uses historical lottery data (`dfrev.csv`) that includes:
- `date`: The date of the lottery draw.
- `firstprize`: The winning lottery number.
- `d1` to `d6`: The individual digits of the lottery number.

### Dataset Columns:
- `date`: Date of the lottery draw.
- `d1` to `d6`: Digits of the lottery number.
- `firstprize`: The winning lottery number.

## 🚀 Useage
- https://huayhack.streamlit.app/
