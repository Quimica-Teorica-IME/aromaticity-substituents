import pandas as pd
import matplotlib.pyplot as plt
import os

# Função para gerar os gráficos e salvar nas pastas Absolut e Normalized
def generate_plots(csv_file, folder_name, normalize=False):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create the respective folder if it doesn't exist
    if not os.path.exists(f'Plots/{folder_name}'):
        os.makedirs(f'Plots/{folder_name}')

    # Get the column names, excluding the 'Molecule' column
    columns = df.columns[1:]

    # Loop through the columns and create plots
    for column in columns:
        # Clean the column name: remove spaces and '|'
        cleaned_column_name = column.replace(' ', '').replace('|', '')

        # Create the bar plot with all bars in lightskyblue
        plt.figure(figsize=(10, 6))

        # Filter out 'C6H6' from the dataframe to avoid plotting it
        df_filtered = df[df['Molecule'] != 'C6H6']

        # Create bars for all molecules excluding C6H6
        bars = plt.bar(df_filtered['Molecule'], df_filtered[column], color='lightskyblue')

        # Get the value of C6H6 for the reference line
        benzene_value = None
        if 'C6H6' in df['Molecule'].values:
            benzene_value = df[df['Molecule'] == 'C6H6'][column].values[0]

        # Add the dashed reference line for 'C6H6' (benzene)
        if benzene_value is not None:
            plt.axhline(y=benzene_value, color='black', linestyle='--', linewidth=1, label='C6H6 Reference')

        # Set the title and labels
        plt.title(f'{column}', fontsize=14)
        plt.xlabel('Molecule', fontsize=12)

        # Modify the ylabel for the Normalized graphs
        if normalize:
            plt.ylabel(f'Normalized {column} (%)', fontsize=12)
        else:
            plt.ylabel(column, fontsize=12)

        # Rotate the molecule names to be horizontal and written from bottom to top
        plt.xticks(rotation=90, ha='center')

        # Display legend to show the reference line
        plt.legend()

        # Save the image in the respective folder
        plt.tight_layout()
        plt.savefig(f'Plots/{folder_name}/{cleaned_column_name}.png')

        # Close the figure to free up memory
        plt.close()

    print(f"Plots generated and saved in the 'Plots/{folder_name}' folder.")

# Generate plots for normalized_results.csv and save in Normalized folder
generate_plots('normalized_results.csv', 'Normalized', normalize=True)

# Generate plots for output_results.csv and save in Absolut folder
generate_plots('output_results.csv', 'Absolut', normalize=False)
