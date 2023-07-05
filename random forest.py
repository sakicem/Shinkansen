X = df_train.drop(['ID', 'Overall_Experience'], axis=1)
y = df_train['Overall_Experience']

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Evaluate accuracy on the validation set
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", val_accuracy)

# Make predictions on the test set
test_predictions = model.predict(df_test.drop('ID', axis=1))

# Create a submission DataFrame
submission = pd.DataFrame({'ID': df_test['ID'], 'Overall_Experience': test_predictions})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)