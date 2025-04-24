import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import re  
from adjustText import adjust_text



# Load CSV files
df_francisco = pd.read_csv("francisco.csv", sep=',')
df_mine = pd.read_csv("mine.csv", sep=',')

# Create the main directory for plots
os.makedirs("plots", exist_ok=True)

# Function to clean column names for filenames (removes problematic characters)
def clean_filename(name):
    return re.sub(r'[|() ]', '', name)  # Remove |, (), spaces


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

# Iterate over each descriptor in Francisco and Mine
for col_f in df_francisco.columns[1:]:  # Ignore first column (X)
    col_f_clean = clean_filename(col_f)
    
    # Create subfolders for all data and filtered data
    folder_path_all_data = os.path.join("plots", col_f_clean, 'all_data')
    os.makedirs(folder_path_all_data, exist_ok=True)
    
    folder_path_filtered = os.path.join("plots", col_f_clean, 'without_CH-2_and_NH+2')
    os.makedirs(folder_path_filtered, exist_ok=True)

    for col_m in df_mine.columns[1:]:  # Ignore first column (X)
        # Prepare data for all data plot
        x = df_francisco[col_f].values.reshape(-1, 1)
        y = df_mine[col_m].values.reshape(-1, 1)

        if len(x) == 0 or len(y) == 0:
            print(f"Not enough data for {col_f} vs {col_m}")
            continue

        # Linear Regression for all data
        model_all = LinearRegression()
        model_all.fit(x, y)
        y_pred_all = model_all.predict(x)
        r2_all = r2_score(y, y_pred_all)
        coef_all, intercept_all = model_all.coef_[0][0], model_all.intercept_[0]


        # Create plot for all data
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=df_francisco[col_f], y=df_mine[col_m], label='Data')
        plt.plot(df_francisco[col_f], y_pred_all, color='red', linestyle='solid', linewidth=0.8,  # Thinner red line
                 label=f'R² = {r2_all:.2f}\nY = {coef_all:.2f}X + {intercept_all:.2f}')
        
        # Set axis labels without adding additional text
        plt.xlabel(col_f, fontsize=16)
        plt.ylabel(col_m, fontsize=16)
        plt.legend(fontsize=14)

        # Add labels to points
        texts = []
        for i, label in enumerate(df_francisco.iloc[:, 0]):  
            formatted_label = format_label(label)
            texts.append(plt.text(x[i], y[i], formatted_label, fontsize=14)) 

        adjust_text(
            texts,
            only_move={'points': 'y', 'text': 'y'},
            expand_text=(1.1, 1.2),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )

        # Generate filename and save plot in the 'all_data' folder
        filename_all = os.path.join(folder_path_all_data, f"{col_f_clean}_vs_{clean_filename(col_m)}.png")
        plt.savefig(filename_all, dpi=300)
        plt.close()

        # Print only the filename (not the full path)
        print(f"Plot saved (all data): {os.path.basename(filename_all)}")

        # Prepare filtered data (remove rows where first column contains 'CH-2' or 'NH+2')
        df_francisco_filtered = df_francisco[~df_francisco.iloc[:, 0].isin(['CH-2', 'NH+2'])]
        df_mine_filtered = df_mine[~df_mine.iloc[:, 0].isin(['CH-2', 'NH+2'])]

        # Linear Regression for filtered data
        x_filtered = df_francisco_filtered[col_f].values.reshape(-1, 1)
        y_filtered = df_mine_filtered[col_m].values.reshape(-1, 1)

        if len(x_filtered) == 0 or len(y_filtered) == 0:
            print(f"Not enough data for {col_f} vs {col_m} after filtering")
            continue

        model_filtered = LinearRegression()
        model_filtered.fit(x_filtered, y_filtered)
        y_pred_filtered = model_filtered.predict(x_filtered)
        r2_filtered = r2_score(y_filtered, y_pred_filtered)
        coef_filtered, intercept_filtered = model_filtered.coef_[0][0], model_filtered.intercept_[0]

        # Create plot for filtered data (without CH-2 and NH+2)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df_francisco_filtered[col_f], y=df_mine_filtered[col_m], label='Data')
        plt.plot(df_francisco_filtered[col_f], y_pred_filtered, color='red', linestyle='solid', linewidth=0.8,  # Thinner red line
                 label=f'R² = {r2_filtered:.2f}\nY = {coef_filtered:.2f}X + {intercept_filtered:.2f}')
        
        # Set axis labels without adding additional text
        plt.xlabel(col_f, fontsize=16)
        plt.ylabel(col_m, fontsize=16)
        plt.legend(fontsize=14)

        # Add labels to filtered points
        texts = []
        for i, label in enumerate(df_francisco_filtered.iloc[:, 0]):
            formatted_label = format_label(label)
            texts.append(plt.text(x_filtered[i], y_filtered[i], formatted_label, fontsize=8))

        adjust_text(
            texts,
            only_move={'points': 'y', 'text': 'y'},
            expand_text=(1.1, 1.2),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )

        # Generate filename and save plot in the 'without_CH-2_and_NH+2' folder
        filename_filtered = os.path.join(folder_path_filtered, f"{col_f_clean}_vs_{clean_filename(col_m)}.png")
        plt.savefig(filename_filtered, dpi=300)
        plt.close()

        # Print only the filename (not the full path)
        print(f"Plot saved (filtered data): {os.path.basename(filename_filtered)}")

print("All plots have been generated and saved inside the 'plots' folder.")
