
from datetime import datetime
import os
import pandas as pd


class CommonUtils:
    @staticmethod
    def save_to_xlsx(data: list[dict], filename: str, base_folder:str):
        """ Save results to Excel file """
        if not data:
            print("No data to save.")
            return
        
        # Create directory structure for saving Excel files
        today_folder = datetime.now().strftime("%Y-%m-%d")
        save_path = os.path.join(base_folder, today_folder)
        
        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Remove .xlsx extension if already present, then add timestamped version
        base_filename = filename.replace('.xlsx', '')
        timestamped_filename = f'{base_filename}_{datetime.now().strftime("%Y%m%d")}.xlsx'
        full_filepath = os.path.join(save_path, timestamped_filename)

        df = pd.DataFrame(data)
        df.to_excel(full_filepath, index=False)
        print(f"Results saved to {full_filepath}")