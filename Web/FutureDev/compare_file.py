import pandas as pd  
import sys  

def compare_csv(file_path1, file_path2):  
    # Read the CSV files  
    try:  
        df1 = pd.read_csv(file_path1)  
        df2 = pd.read_csv(file_path2)  
    except Exception as e:  
        print(f"Error reading files: {e}")  
        return  

    # Compare the DataFrames  
    if df1.equals(df2):  
        print("The two CSV files are identical.")  
    else:  
        print("The two CSV files are different.")  
        
        # You can show the differences if needed  
        differences = pd.concat([df1, df2]).drop_duplicates(keep=False)  
        print("Differences:")  
        print(differences)  

if __name__ == "__main__":  
    # if len(sys.argv) != 3:  
    #     print("Usage: python compare_csv.py <path_to_csv1> <path_to_csv2>")  
    # else:  
    for i in range(1, 30): 
        print("-", i) 
        csv_file_path1 = f"D:\cd_data_C\Downloads\submit (2)\query-p2-{i}-kis.csv"
        csv_file_path2 = f"D:\cd_data_C\Downloads\submit (4)\query-p2-{i}-kis.csv" 
        compare_csv(csv_file_path1, csv_file_path2)