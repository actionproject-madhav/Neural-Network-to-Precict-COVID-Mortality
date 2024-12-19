import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Read the data
df = pd.read_csv("Covid Data.csv")

# Replace some random values seen in data
df[df.columns.drop("AGE")] = df[df.columns.drop("AGE")].replace([97, 98, 99], np.nan)

# Convert death data to binary where 1 means died and 0 means not
df["DIED"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)

# Drop the original column of death data
df = df.drop("DATE_DIED", axis=1)

# Handle missing data
df.loc[df["PATIENT_TYPE"] == 1, ["INTUBED", "ICU"]] = 2
df.loc[(df["INTUBED"] == 2) & (df["ICU"].isna()), "ICU"] = 2

df.loc[df["PATIENT_TYPE"] == 1, "PNEUMONIA"] = df.loc[df["PATIENT_TYPE"] == 1, "PNEUMONIA"].fillna(2)

# Handling Missing Values for Pregnant Feature
df.loc[df["SEX"] == 2, "PREGNANT"] = 2
df.loc[(df["AGE"] >= 50) & (df["SEX"] == 1), "PREGNANT"] = 2
df.loc[(df["PREGNANT"].isna()) & (df["AGE"] >= 50), "PREGNANT"] = 2

# Handling Missing Values for Diabetes
df.loc[df["AGE"] < 30, "DIABETES"] = df.loc[df["AGE"] < 30, "DIABETES"].fillna(2)

df.loc[(df["AGE"].between(30, 40)) & (df["HIPERTENSION"] == 2), "DIABETES"].fillna(2, inplace=True)
df.loc[df["AGE"] < 30, "HIPERTENSION"] = df.loc[df["AGE"] < 30, "HIPERTENSION"].fillna(2)

# Drop remaining missing values
df = df.dropna()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=df.columns.drop(["AGE", "DIED"]), drop_first=True)

# Split data into features and labels
X_train, X_test, y_train, y_test = train_test_split(df.drop("DIED", axis=1), df["DIED"], test_size=0.3, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Model
model = Sequential()

# Input Layer - First hidden layer
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Second hidden layer
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output Layer (binary classification - 1 or 0)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping])

# Plot training and validation loss/accuracy
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Survived", "Died"], yticklabels=["Survived", "Died"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random guess line
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
