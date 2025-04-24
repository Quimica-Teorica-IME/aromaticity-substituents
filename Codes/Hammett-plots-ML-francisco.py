import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
import numpy as np
from adjustText import adjust_text

# File path
file_path = "hammettxfrancisco.csv"

# Load the CSV file with separator ","
try:
    data = pd.read_csv(file_path, sep=",", encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv(file_path, sep=",", encoding="latin1")

# Diagnose column names
print("Detected columns:", data.columns)

# Remove extra spaces from columns
data.columns = data.columns.str.strip()

# Check if the columns are in the file
sigma_columns = ['σm', 'σm0', 'σR', 'σI', 'σp', 'σp0', 'σp+', 'σp-']
Francisco_columns = ['HOMA','AI(vib)','BVI','MCI','NICS(0)','NICS(1)','NICS(1)zz']

# Create directory to save graphs
output_dir = "graphs_hammettxFrancisco_ML"
os.makedirs(output_dir, exist_ok=True)

# Function to clean up filenames and directory names
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# Function to format labels with numbers as subscripts and symbols + or - as superscripts
def format_label(label):
    # Replace numbers with subscripts while keeping the rest of the text
    label = re.sub(r'(\d+)', r'$_{\1}$', label)
    
    # Check if the label ends with + or -
    if label.endswith('+'):
        label = label[:-1] + r'$\mathregular{^{+}}$'  # Format the + symbol as superscript
    elif label.endswith('-'):
        label = label[:-1] + r'$\mathregular{^{-}}$'  # Format the - symbol as superscript
    
    return label

# Loop to create graphs
for sigma in sigma_columns:
    for F in Francisco_columns:
        if sigma in data.columns and F in data.columns:
            subset = data[['-X', sigma, F]].dropna()  # Include the -X column in the subset
            if not subset.empty:
                x = subset[sigma].values.reshape(-1, 1)  # Convert to the proper format for regression
                y = subset[F].values
                
                # Linear regression (linear fit)
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)
                
                # Line coefficients
                slope = model.coef_[0]
                intercept = model.intercept_
                r2 = model.score(x, y)
                
                # Clustering with DBSCAN
                clustering = DBSCAN(eps=0.7, min_samples=3).fit(subset[[sigma, F]])  # DBSCAN with adjusted eps and min_samples
                labels = clustering.labels_
                unique_labels = np.unique(labels)

                # Create a folder for each Francisco_column inside the main output directory
                Francisco_cleaned = sanitize_filename(F)  # Clean the Francisco_column name for use as folder name
                Francisco_output_dir = os.path.join(output_dir, Francisco_cleaned)
                os.makedirs(Francisco_output_dir, exist_ok=True)  # Create the folder if it doesn't exist

                # Scatter plot
                plt.figure(figsize=(10, 8))  # Increase the figure size
                for label in unique_labels:
                    # Filter points according to the cluster
                    cluster_data = subset[labels == label]
                    if label == -1:  # If DBSCAN marks as "noise"
                        label = 'Noise'
                    plt.scatter(cluster_data[sigma], cluster_data[F], label=f'Cluster {label+1}' if label != 'Noise' else label, edgecolor='black', alpha=0.7, s=400)  # Add each cluster with a different color
                plt.xlabel(sigma, fontsize=25)  # Increase the size of the X-axis label
                plt.ylabel(F, fontsize=25)  # Increase the size of the Y-axis label
                plt.xticks(fontsize=25)  # Increase the size of the X-axis values
                plt.yticks(fontsize=25)  # Increase the size of the Y-axis values
                plt.grid(True)
                
                # Add labels to points with automatic adjustment
                texts = []
                for i, label in enumerate(subset['-X']):
                    # Format the label to add subscripts and superscripts
                    formatted_label = format_label(label)
                    texts.append(plt.text(x[i], y[i], formatted_label, fontsize=25))  # Triple the font size of the labels
                
                # Adjust the labels to avoid overlap and add arrows
                adjust_text(
                    texts,
                    only_move={'points': 'y', 'text': 'y'},  # Move only on the Y-axis
                    expand_text=(1.1, 1.2),  # Expand the available space for labels
                    arrowprops=dict(arrowstyle="->", color="black", lw=2),  # Arrows are now thicker and black
                )
                
                # Check if R2 is greater than or equal to 0.6 to plot the line
                if r2 >= 0.6:
                    # Plot the regression line (fitted line) - dotted line with larger spacing
                    plt.plot(x, y_pred, color='red', linestyle='--', linewidth=2, dashes=(1, 8), label=f'{slope:.2f}x + {intercept:.2f}')
                
                    # Create the equation and R² as a string
                    equation_text = f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r2:.3f}$'
                    
                    # Add the equation and R² in the bottom left corner unless it is a special case
                    #if not (sigma in ['σp', 'σp-', 'σp0', 'σI']) and F == 'BVI':
                        #plt.text(0.05, 0.05, equation_text, transform=plt.gca().transAxes, fontsize=20,
                        #         verticalalignment='bottom', horizontalalignment='left', color='black',
                        #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                
                # Check if the graph should have labels in the bottom right corner
                #if sigma in ['σp', 'σp-', 'σp0', 'σI'] and F == 'BVI':
                    # Add the equation and R² in the bottom right corner (not the bottom left corner)
                    #equation_text = f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r2:.3f}$'
                    #plt.text(0.95, 0.05, equation_text, transform=plt.gca().transAxes, fontsize=20,
                    #        verticalalignment='bottom', horizontalalignment='left', color='black',
                    #        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                
                # Remove the title of the graph
                plt.title("")  # Do not place the title "Relationship between..."
                
                # Save the graph
                plt.tight_layout()
                scatter_filename = sanitize_filename(f"scatter_{F}_vs_{sigma}.png")
                plt.savefig(os.path.join(Francisco_output_dir, scatter_filename))
                plt.close()

print(f"Scatter plots generated and saved in the '{output_dir}' folder, organized by Francisco_columns.")
