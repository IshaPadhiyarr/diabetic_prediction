# -*- coding: utf-8 -*-
"""app.py - Streamlit Optimized Version"""

# Import the necessary libraries
import streamlit as st
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
import numpy 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from numpy import set_printoptions
from pickle import dump
from pickle import load


# Set the title for the Streamlit app
st.title("Diabetic Prediction Model Analysis (Pima Indians)")
st.write("This app runs a machine learning pipeline for diabetic prediction.")

# --- Load the data ---
st.header("1. Data Loading and Shape")
try:
    # Load CSV using Pandas. Assumes the file 'pima-indians-diabetes.csv'
    # is available in the deployment folder.
    filename = 'pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    st.write("Data loaded successfully.")
    st.write("Data Shape:", data.shape)
except FileNotFoundError:
    st.error(f"Error: The data file '{filename}' was not found. Ensure it is deployed with the app.")
    st.stop()

# ----------------------------------------------------------------------------------------------------

# --- Descriptive Stats ---
st.header("2. Descriptive Statistics")

# Data Types for Each Attribute
types = data.dtypes
st.subheader("Data Types")
# Fix: Convert Series to DataFrame for robust st.dataframe() rendering
st.dataframe(pd.DataFrame(types, columns=['Data Type'])) 

# Statistical Summary
st.subheader("Statistical Summary")
st.dataframe(data.describe())

# Pairwise Pearson correlations
correlations = data.corr(method='pearson')
st.subheader("Pairwise Pearson Correlations")
st.dataframe(correlations)

# Class proportion
class_counts = data.groupby('class').size()
st.subheader("Class Proportion")
st.write(class_counts)

# ----------------------------------------------------------------------------------------------------

# --- Data Visualization ---
st.header("3. Data Visualization")

# Histograms
st.subheader("Histograms")
# Fix: Correct Matplotlib Figure handling (use subplots, close after st.pyplot)
fig_hist, axes = pyplot.subplots(3, 3, figsize=(10, 8)) 
data.hist(ax=axes.flatten())
st.pyplot(fig_hist) 
pyplot.close(fig_hist) 

# Density Plots
st.subheader("Density Plots")
# Fix: Correct Matplotlib Figure handling
fig_density, axes = pyplot.subplots(3, 3, figsize=(10, 8))
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, ax=axes.flatten())
st.pyplot(fig_density) 
pyplot.close(fig_density) 

# Box and Whisker Plots
st.subheader("Box and Whisker Plots")
# Fix: Correct Matplotlib Figure handling
fig_box, axes = pyplot.subplots(3, 3, figsize=(10, 8))
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, ax=axes.flatten())
st.pyplot(fig_box) 
pyplot.close(fig_box) 

# Correction Matrix Plot
st.subheader("Correction Matrix Plot")
# Ensure standard figure creation and closing sequence
fig_corr = pyplot.figure()
ax = fig_corr.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig_corr.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
st.pyplot(fig_corr)
pyplot.close(fig_corr)

# ----------------------------------------------------------------------------------------------------

# --- Data Preprocessing ---
st.header("4. Data Preprocessing (Standardization)")
# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]
# Standardize data
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
st.write("Standardized Data (First 5 rows):")
st.code(str(rescaledX[:5,:]))

# ----------------------------------------------------------------------------------------------------

# --- Algorithm Evaluation ---
st.header("5. Algorithm Evaluation (10-Fold Cross-Validation)")

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'

# Prepare models
models = []
models.append(('LR', LogisticRegression(max_iter=200)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
performance_data = [] # List to store results for a DataFrame

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, rescaledX, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    mean_score = cv_results.mean()
    std_score = cv_results.std()
    
    # Capturing output for display
    performance_data.append({
        'Model': name,
        'Mean Accuracy': f"{mean_score:f}",
        'Std Dev': f"({std_score:f})"
    })

# Display algorithm performance
st.subheader("Model Performance Summary")
st.dataframe(pd.DataFrame(performance_data).set_index('Model'))

# Compare Algorithms
st.subheader("Algorithm Comparison (Box Plot)")
fig_comp = pyplot.figure()
fig_comp.suptitle('Algorithm Comparison')
ax = fig_comp.add_subplot(111)
pyplot.boxplot(results, labels=names)
st.pyplot(fig_comp)
pyplot.close(fig_comp)

# ----------------------------------------------------------------------------------------------------

# --- Algorithm Tuning ---
st.header("6. Algorithm Tuning (Grid Search on SVM)")

# Grid Search on SVM
# (Code remains the same for computation)
C_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=C_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y)

st.subheader("Best Parameters Found")
st.write("Best Score: **%f** using parameters: `%s`" % (grid_result.best_score_, grid_result.best_params_))

# Display all results in a table
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
grid_results_df = pd.DataFrame({
    'Mean Score': [f"{m:f}" for m in means],
    'Std Dev': [f"({s:f})" for s in stds],
    'Parameters': [str(p) for p in params]
})
st.subheader("Detailed Grid Search Results")
st.dataframe(grid_results_df)

# ----------------------------------------------------------------------------------------------------

# --- Finalize Model ---
st.header("7. Finalize and Save Model")

# Split array into input and output
X = array[:,0:8]
Y = array[:,8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# Fit the model on 33%
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# save the model to disk
filename_save = 'finalized_model.sav'
try:
    dump(model, open(filename_save, 'wb'))
    st.success(f"Model saved to disk as '{filename_save}'")
except Exception as e:
    st.warning(f"Could not save model to disk. Error: {e}")

# Load the model and check score
st.subheader("Model Loading and Test Score")
try:
    loaded_model = load(open(filename_save, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    st.write(f"Test Accuracy: **{result:.3f}**")
except FileNotFoundError:
    st.warning(f"Could not load the model from disk (file not found).")
    
# Clean up Matplotlib figures
pyplot.close('all')
