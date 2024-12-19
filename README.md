# COVID-19 Mortality Predictions Using Neural Network with Above 95% Accuracy
 Neural network to predict patient mortality using real data, feature scaling and performance evaluation

# Data
Data from 1M+ users was taken from https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data. The dataset contains 20 features and mortality information. There was much-missing data on diabetes, pregnancy and so on. The columns were modified to be usable for the neural network; for instance, the death information column was converted to a binary column with 0 meaning didn't die and 1 meaning died. The missing data was carefully estimated using correlations with other features and common sense. For example, 50+ aged Women and all men were set to False under Pregnancy, people aged below 30 were set to False for Diabetes and so on. The dataset was left with a few empty cells which was removed as it wouldn't have much significant impact on overall results.

# Neural Network
A fully connected feedforward Neural Network was built and trained on 1M+ patient data. The network has one input, two hidden, and one output layer. The two hidden layers have ReLU activation, batch normalization and dropout for regularization. The output layer is a single neuron with sigmoid activation for binary classification. 
![neural_network_architecture](https://github.com/user-attachments/assets/6b6e4079-ecbe-4799-b828-13b83d42b896)

# Refining the Model
The model was experimented with and trained with different layers, activations, and algorithms like random forest. The above neural network was the final output with feature scaling, hyperparameter tuning, and regularization like dropout. It was trained for 100 epochs for better accuracy .It has an accuracy above 95% and does a fine job with test data.

# Plots
![Training and Validation](https://github.com/user-attachments/assets/87d4451c-7c20-4ae2-a483-bb88e922f093)
![Confusion Matrix](https://github.com/user-attachments/assets/0a9d9d0f-63ba-4d68-ad0c-c54e861dda81)







