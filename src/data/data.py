import os 
import pandas as pd
import json
folder_path = os.path.dirname(os.path.abspath(__file__))

def save_data(df, num):
    """
    Save the DataFrame to a CSV file with the specified number.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    num (str): The number to append to the filename.
    """
    file_path = os.path.join(folder_path, 'AITEX_' + num + '.csv')
    
    if num == "00":
        raise ValueError("Cannot save data with num '00'. Use a different number.")
    
    df.to_csv(file_path, index=False)
    
    print(f"Data saved to {file_path}")

def load_data(num="00"):
    """
    Load the DataFrame from a CSV file with the specified number.

    Parameters:
    num (str): The number to append to the filename.
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """

    df =  pd.read_csv(os.path.join(folder_path, 'AITEX_'+num+'.csv'))

    if num == "00":
        df.pop("Test")
        df.pop("Date")
    return df

def load_physical_data():
    """
    Load the physical properties data from a CSV file.
    
    Returns:
    pd.DataFrame: The DataFrame containing physical properties.
    """
    file_path = os.path.join(folder_path, 'PHYSICAL.csv')
    df = pd.read_csv(file_path)
    
    # Rename columns to match expected format
    df.columns = ['Orientation (deg)', 'Crystallinity (%)', 'Strength (MPa)']
    
    return df


def load_density_data():
    """
    Load the density data from a CSV file.
    
    Returns:
    pd.DataFrame: The DataFrame containing density data.
    """
    file_path = os.path.join(folder_path, 'DENSITY.json')
    
    df = json.load(open(file_path, 'r'))
    return df