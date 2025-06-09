import os 
import pandas as pd
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