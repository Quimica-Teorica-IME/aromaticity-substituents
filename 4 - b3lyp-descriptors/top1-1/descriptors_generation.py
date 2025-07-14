import os
import re
import pandas as pd
import glob

def calculate_q2_ring_atoms(lines, start_index, end_index):
    """
    Function to calculate the sum of |Q2| values for ring atoms.
    Returns the sum of the found values or None if fewer than 6 values are found.
    """
    q2_ring_values = []
    for i in range(start_index, end_index):
        line = lines[i]
        if "C" in line:  # Only carbon atoms
            # Search for the value of |Q2| after finding "C"
            for j in range(i + 1, len(lines)):
                if "|Q2|" in lines[j]:
                    parts = lines[j].split('=')
                    if len(parts) > 1:
                        q2_value = parts[1].split()[0].strip()  # Extract the number
                        try:
                            q2_ring_values.append(float(q2_value))
                        except ValueError:
                            print(f"Error converting |Q2| to float: {q2_value}")
                    if len(q2_ring_values) == 6:  # Stop after finding 6 values
                        return sum(q2_ring_values)  # Return the sum of values
    print("Less than 6 |Q2| values found.")
    return None


def calculate_qzz_ring_atoms(lines, start_index, end_index):
    """
    Function to calculate the sum of Qzz=Q20 values for ring atoms.
    Returns the sum of the found values or None if fewer than 6 values are found.
    """
    qzz_ring_values = []
    for i in range(start_index, end_index):
        line = lines[i]
        if "C" in line:  # Only carbon atoms
            # Search for the value of Qzz = Q20 after finding "C"
            for j in range(i + 1, len(lines)):
                if "Q20" in lines[j]:
                    content_after_qzz = lines[j].split("Q20")[1]  # Capture everything after "Q20"
                    parts = content_after_qzz.split('=')
                    if len(parts) > 1:
                        qzz_value = parts[1].split()[0].strip()  # Extract the number
                        try:
                            qzz_ring_values.append(float(qzz_value))
                        except ValueError:
                            print(f"Error converting Qzz = Q20 to float: {qzz_value}")
                    if len(qzz_ring_values) == 6:  # Stop after finding 6 values
                        return sum(qzz_ring_values)  # Return the sum of values
    print(f"Less than 6 Qzz = Q20 values found.")
    return None


def calculate_q2_origin(lines):
    """
    Function to calculate the |Q2| value for the origin.
    Returns the value found or None if no value is found.
    """
    q2_origin = None
    for i, line in enumerate(lines):
        if 'Total multipoles referred to origin' in line:
            # Search for the values after the line 'Total multipoles referred to origin'
            for j in range(i + 1, len(lines)):
                if '|Q2|' in lines[j]:
                    parts = lines[j].split('=')
                    if len(parts) > 1:
                        q2_origin = parts[1].split()[0].strip()  # Extract the number
                        try:
                            q2_origin = float(q2_origin)
                        except ValueError:
                            print(f"Error converting |Q2| origin to float: {q2_origin}")
                    break
            break
    
    if q2_origin is None:
        print("No |Q2| origin value found.")
    return q2_origin

def calculate_qzz_origin(lines):
    """
    Function to calculate the Qzz (Q20) value for the origin.
    Returns the value found or None if no value is found.
    """
    qzz_origin = None
    for i, line in enumerate(lines):
        if 'Total multipoles referred to origin' in line:
            # Search for the values after the line 'Total multipoles referred to origin'
            for j in range(i + 1, len(lines)):
                if 'Q20' in lines[j]:
                    content_after_qzz = lines[j].split("Q20")[1]  # Capture everything after "Q20"
                    parts = content_after_qzz.split('=')
                    if len(parts) > 1:
                        qzz_value = parts[1].split()[0].strip()  # Extract the number
                        try:
                            qzz_origin = float(qzz_value)
                        except ValueError:
                            print(f"Error converting Qzz origin to float: {qzz_value}")
                    break
            break
    
    if qzz_origin is None:
        print("No Qzz origin value found.")
    return qzz_origin


def calculate_tops(lines):
    """
    Function to calculate the values of |Q2|(n) and Q2(n)zz for all TOPs found.
    Returns a dictionary with the found values.
    """
    tops_data = {}
    for i, line in enumerate(lines):
        # Use regex to find TOP(n) or TOPn, where n can be a number or a letter
        match = re.search(r"TOP(?:\(([\w\-]+)\)|([\w\-]+))", line)
        if match:
            # Extract the TOP identifier (first group captures TOP(n), second group captures TOPn)
            top_id = match.group(1) or match.group(2)
            
            # Search for the |Q2| and Q20 values in the following lines
            q2_value = None
            q20_value = None
            for j in range(i + 1, i + 7):  # Considers up to 6 lines after the TOP line
                if j >= len(lines):
                    break

                # Search for |Q2|
                if '|Q2|' in lines[j]:
                    content_after_q2 = lines[j].split('|Q2|')[1]  # Capture everything after '|Q2|'
                    parts = content_after_q2.split('=')
                    if len(parts) > 1:
                        q2_str = parts[1].split()[0].strip()  # Extract the numeric value
                        try:
                            q2_value = float(q2_str)
                        except ValueError:
                            print(f"Error converting |Q2| to float in TOP({top_id}): {q2_str}")
                 

                # Search for Q20
                if 'Q20' in lines[j]:
                    content_after_q20 = lines[j].split('Q20')[1]  # Capture everything after 'Q20'
                    parts = content_after_q20.split('=')
                    if len(parts) > 1:
                        q20_str = parts[1].split()[0].strip()  # Extract the numeric value
                        try:
                            q20_value = float(q20_str)
                        except ValueError:
                            print(f"Error converting Q20 to float in TOP({top_id}): {q20_str}")


            # Store the values in the dictionary
            if q2_value is not None and q20_value is not None:
                tops_data[top_id] = {"|Q2|": q2_value, "Q20": q20_value}
    
    return tops_data


def process_output_file(output_file, folder_name):
    try:
        with open(output_file, 'r') as file:
            lines = file.readlines()
        
        start_marker = 'Multipole moments in atomic units, ea_0^k for rank k'
        end_marker = 'CPU time used'
        
        # Find the positions of the markers
        start_index = next((i + 2 for i, line in enumerate(lines) if start_marker in line), None)
        end_index = next((i for i in range(start_index, len(lines)) if end_marker in lines[i]), None) if start_index else None
        
        if start_index is None or end_index is None:
            print(f"Error processing {output_file}: Markers not found correctly.")
            return
        
        # Calculate values
        total_q2_ring = calculate_q2_ring_atoms(lines, start_index, end_index)
        total_qzz_ring = calculate_qzz_ring_atoms(lines, start_index, end_index)
        total_q2_origin = calculate_q2_origin(lines)
        total_qzz_origin = calculate_qzz_origin(lines)
        tops_data = calculate_tops(lines)
        
        # Create the result dictionary
        result = {
            'Molecule': folder_name,
            'Total |Q2| (ring atoms)': total_q2_ring,
            'Total Qzz (ring atoms)': total_qzz_ring,
            '|Q2| origin': total_q2_origin,
            'Qzz origin': total_qzz_origin,
        }

        if tops_data:
            for top_id, values in tops_data.items():
                result[f'|Q2|({top_id})'] = values['|Q2|']
                result[f'Q2({top_id})zz'] = values['Q20']
        else:
            print('No additional sites specified')

        # Display results
        print(f"Results for folder '{folder_name}':")
        for label, value in result.items():
            print(f"{label}: {value}")
        
        print("-" * 50 + "\n")
        
        return result

    except Exception as e:
        print(f"Error processing {output_file}: {e}")
        return None


def save_to_csv(results, csv_filename):
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to: {csv_filename}")

def normalizer(df):
    # Create a copy of the dataframe to store normalized values
    df_normalized = df.copy()

    # Loop through the columns (excluding 'Molecule') to normalize each one
    for column in df.columns[1:]:  # Skip the 'Molecule' column
        # Get the value of C6H6 for the current column
        if 'C6H6' in df['Molecule'].values:
            benzene_value = df[df['Molecule'] == 'C6H6'][column].values[0]
            
            # Normalize the column values (divide by benzene value and multiply by 100)
            df_normalized[column] = df[column] / benzene_value * 100

    # Save the normalized results to a new CSV file
    df_normalized.to_csv('normalized_results.csv', index=False)

    print("Normalized results saved to 'normalized_results.csv'.")



    
# Main function
def main():
    root_directory = "."  # Root directory
    all_results = []  # List to store the results

    # Traverse all folders in the root directory
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        
        if os.path.isdir(folder_path):
            output_files = glob.glob(os.path.join(folder_path, "*gdma*.output"))
            
            if output_files:
                for output_file in output_files:
                    result = process_output_file(output_file, folder_name)
                    
                    # If results are found, add them to the list
                    if result:
                        all_results.append(result)
            else:
                print(f"No 'gdma*.output' file found in folder: {folder_path}")
    
    if all_results:
        # Save the results to a CSV
        save_to_csv(all_results, "output_results.csv")
        dataframe = pd.DataFrame(all_results)
        
        normalizer(dataframe)
    else:
        print("No results found to save.")
    
    
if __name__ == "__main__":
    main()
