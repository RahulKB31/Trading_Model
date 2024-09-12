import numpy as np
from sklearn.model_selection import train_test_split
import read_vectors as rv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Convert vectors into numpy arrays
vectors = np.array([rv.word_vectors[key] for key in rv.metadata])

# Define target variable
# Using the difference between consecutive vectors
y = np.array([np.sum(vectors[i+1]) - np.sum(vectors[i]) for i in range(len(vectors)-1)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(vectors[:-1], y, test_size=0.2, shuffle=False)

# Reshape data for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build and train LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Predict price movements with LSTM
y_pred_lstm = model.predict(X_test)

# Define trading strategy based on LSTM predictions
def trading_strategy(pred):
    if pred > 0.1:
        return "Buy"
    elif pred < -0.1:
        return "Sell"
    else:
        return "Hold"

# Apply strategy to LSTM predictions
actions_lstm = [trading_strategy(pred) for pred in y_pred_lstm]
print("Actions based on LSTM predictions:", actions_lstm)

# Convert continuous target to categorical classes for Random Forest
y_class = np.where(y > 0.1, 1, np.where(y < -0.1, -1, 0))

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train.reshape((X_train.shape[0], X_train.shape[2])), y_class[:len(X_train)])

# Predict and evaluate Random Forest
y_pred_class = clf.predict(X_test.reshape((X_test.shape[0], X_test.shape[2])))
accuracy_rf = accuracy_score(y_class[len(X_train):], y_pred_class)

# Print Random Forest Accuracy
print(f"Random Forest Accuracy: {accuracy_rf}")

# Example thresholds and prices for Risk Management
stop_loss = 0.02  # 2% stop-loss
take_profit = 0.05  # 5% take-profit
entry_price = 100  # Example entry price
current_prices = [101, 99, 102, 95]  # Example current prices

# Implement Risk Management based on LSTM predictions
def apply_risk_management(action, current_price, entry_price, stop_loss, take_profit):
    if entry_price == 0:
        return "Hold"

    if action == "Buy":
        if (current_price - entry_price) >= take_profit:
            return "Exit (Take Profit)"
        elif (current_price - entry_price) <= -stop_loss:
            return "Exit (Stop Loss)"
    elif action == "Sell":
        if (entry_price - current_price) >= take_profit:
            return "Exit (Take Profit)"
        elif (entry_price - current_price) <= -stop_loss:
            return "Exit (Stop Loss)"

    return action

# Example actions based on Random Forest predictions
actions_rf = [trading_strategy(pred) for pred in clf.predict(X_test.reshape((X_test.shape[0], X_test.shape[2])))]
print("Random Forest Actions based on predictions:", actions_rf)

# Apply risk management to example prices
for current_price in current_prices:
    action_lstm = apply_risk_management("Buy", current_price, entry_price, stop_loss, take_profit)
    print(f"LSTM Action: {action_lstm}, Price: {current_price}")

    action_rf = apply_risk_management("Buy", current_price, entry_price, stop_loss, take_profit)
    print(f"Random Forest Action: {action_rf}, Price: {current_price}")
