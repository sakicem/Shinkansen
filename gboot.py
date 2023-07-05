
target = 'Overall_Experience'
features = ['Gender', 'Customer_Type', 'Age', 'Type_Travel', 'Travel_Class', 'Travel_Distance',
            'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins', 'Seat_Comfort', 'Seat_Class',
            'Arrival_Time_Convenient', 'Catering', 'Platform_Location', 'Onboard_Wifi_Service',
            'Onboard_Entertainment', 'Online_Support', 'Ease_of_Online_Booking', 'Onboard_Service',
            'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Cleanliness', 'Online_Boarding']

X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train[target], test_size=0.2, random_state=42)

params = {
    'objective': 'binary:logistic',  # Binary classification
    'eta': 0.1,  # Learning rate
    'max_depth': 3,  # Maximum depth of each tree
    'n_estimators': 100  # Number of trees (boosting rounds)
}

# Create the XGBoost Classifier
model = xgb.XGBClassifier(**params)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

predictions = model.predict(df_test[features])

# Create a DataFrame with the ID and Overall_Experience predictions
output = pd.DataFrame({'ID': df_test['ID'], 'Overall_Experience': predictions})

# Save the predictions to a CSV file
output.to_csv('predictions.csv', index=False)