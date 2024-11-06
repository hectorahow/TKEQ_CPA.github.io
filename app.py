from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the data
data = pd.read_csv('tkeq_ad_data_total.csv')
metrics = ['Date', 'purchases', 'spend']
data = data[metrics]
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Model setup
x = data[['spend']]
y = data['purchases']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = np.floor(mean_squared_error(y_test, y_pred))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    spend_value = float(request.json['spend_value'])  # Updated to read JSON input
    predicted_purchases = int(np.ceil(model.predict([[spend_value]])[0]))
    cpa1 = (spend_value / predicted_purchases - mse)
    cpa2 = (spend_value / predicted_purchases + mse)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='spend', y='purchases', color='blue', label='Actual Data')
    sns.lineplot(x=x_test['spend'], y=y_pred, color='red', label='Regression Line')
    plt.scatter([spend_value], [predicted_purchases], color='green', marker='X', s=100, label='Predicted Purchases')
    plt.title('Spend vs Purchases with Linear Regression Line and Predicted Points')
    plt.xlabel('Spend')
    plt.ylabel('Purchases')
    plt.legend()

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    result_text = f'{predicted_purchases} ± {mse:.0f} purchases and a CPA between ${cpa1:.1f} - ${cpa2:.1f} are forecast for a spend of ${spend_value}'

    return jsonify(result_text=result_text, plot_url=plot_url)

if __name__ == '__main__':
    app.run()
