# deep-learning-challenge
---
# Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received access to a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
  - EIN and NAME—Identification columns
  - APPLICATION_TYPE—Alphabet Soup application type
  - AFFILIATION—Affiliated sector of industry
  - CLASSIFICATION—Government organization classification
  - USE_CASE—Use case for funding
  - ORGANIZATION—Organization type
  - STATUS—Active status
  - INCOME_AMT—Income classification
  - SPECIAL_CONSIDERATIONS—Special considerations for application
  - ASK_AMT—Funding amount requested
  - IS_SUCCESSFUL—Was the money used effectively

---

# Before You Begin
1. Create a new repository for this project called deep-learning-challenge. Do not add this Challenge to an existing repository.
2. Clone the new repository to your computer.
3. Inside your local git repository, create a directory for the Deep Learning Challenge.
4. Push the above changes to GitHub.

--- 

# Files
Download the following files to help you get started:

[Starter Code](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/Starter_Code.zip)

---

# Instructions

# Step 1: Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

  1. From the provided cloud URL, read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
      - What variable(s) are the target(s) for your model?
      - What variable(s) are the feature(s) for your model?
  2. Drop the EIN and NAME columns.
  3. Determine the number of unique values for each column.
  4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
  5. Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.
  6. Use pd.get_dummies() to encode categorical variables.
  7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
  8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

# Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

  1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
  2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
  3. Create the first hidden layer and choose an appropriate activation function.
  4. If necessary, add a second hidden layer with an appropriate activation function.
  5. Create an output layer with an appropriate activation function.
  6. Check the structure of the model.
  7. Compile and train the model.
  8. Create a callback that saves the model's weights every five epochs.
  9. Evaluate the model using the test data to determine the loss and accuracy.
  10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

# Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
  - Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
    - Add more neurons to a hidden layer.
    - Add more hidden layers.
    - Use different activation functions for the hidden layers.
    - Add or reduce the number of epochs to the training regimen.

## Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

  1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
  2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame from the provided cloud URL.
  3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
  4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
  5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

# Step 4: Write a Report on the Neural Network Model

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
The report should contain the following:

  1. **Overview** of the analysis: Explain the purpose of this analysis.

  2. **Results**: Using bulleted lists and images to support your answers, address the following questions:
      - Data Preprocessing
        - What variable(s) are the target(s) for your model?
        - What variable(s) are the features for your model?
        - What variable(s) should be removed from the input data because they are neither targets nor features?
      
      - Compiling, Training, and Evaluating the Model
        - How many neurons, layers, and activation functions did you select for your neural network model, and why?
        - Were you able to achieve the target model performance?
        - What steps did you take in your attempts to increase model performance?

  3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

# Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.
  1. Download your Colab notebooks to your computer.
  2. Move them into your Deep Learning Challenge directory in your local repository.
  3. Push the added files to GitHub.

-----------------------------------------

# Alphabet Soup Charity Deep Learning Project
---
# Overview

Alphabet Soup is a nonprofit that gives funding to different organizations, and they’re looking for a better way to choose who to support. My goal with this project was to build a deep learning model that can predict if an organization will be successful after getting funded. I used a dataset with info from over 34,000 past applications and built a binary classifier using TensorFlow and Keras.

---

# Data Preprocessing
  - Target: The column I was trying to predict is IS_SUCCESSFUL, which tells us if the organization made good use of the funding.
  - Features: All the other columns — like application type, income level, amount requested, and so on — were used to help the model make a prediction.
  - Removed Columns: I dropped EIN and NAME because they’re just ID fields and don’t add any real value to the model.

Other preprocessing steps:
  - I looked at how many unique values were in each column.
  - If a column had a bunch of rare categories, I grouped those smaller ones into a new “Other” category.
  - I used pd.get_dummies() to convert the categorical columns into numbers the model can understand.
  - Then I split the data into training and test sets and scaled the features using StandardScaler() so they’d be on the same range.

---

# Model Setup and Training
Here’s how I built the model:
- First hidden layer: 256 neurons, ReLU activation
- Second hidden layer: 128 neurons, ReLU activation
- Third hidden layer: 64 neurons, ReLU activation
- Output layer: 1 neuron with a Sigmoid activation since this is a binary classification problem

I compiled the model using the Adam optimizer and binary crossentropy as the loss function.
I trained it for 150 epochs with a batch size of 32.

The final test accuracy came out to about 75%, which meets the target for the project.

---

# Model Optimization
To try and improve the model’s performance, I made a few changes:
  - I added more neurons to each layer than the original version.
  - I included a third hidden layer to help the model learn more detailed patterns.
  - I increased the number of training epochs to give the model more time to improve.

I thought about trying different activation or loss functions, but the ReLU + binary crossentropy combo gave the best results in my early testing, so I stuck with those.

---

# Summary
My final model reached about 75% accuracy, which hit the goal. It does a solid job predicting whether an applicant is likely to be successful, and it could definitely help Alphabet Soup make better funding decisions going forward.

---

# Recommendation
If we wanted to try a different kind of model, I’d suggest testing out a Random Forest or XGBoost model. These are great for structured data like this, usually require less fine-tuning, and can also show which features are the most important. That could give Alphabet Soup better insights into what really drives success.
