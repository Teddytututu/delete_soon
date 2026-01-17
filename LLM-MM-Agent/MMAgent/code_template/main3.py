# ============================================================================
# P0-A FIX: Unified Column Name Standardization
# ============================================================================
# CRITICAL: ALL COLUMN NAMES ARE UPPERCASE (enforced by data loading)
#
# When loading CSV files, use:
#   from utils.data_manager import load_csv_with_standardized_columns
#   df = load_csv_with_standardized_columns('filename.csv')
#
# This automatically:
#   1. Removes BOM characters (ï»¿, Ï»¿)
#   2. Strips whitespace
#   3. Converts ALL column names to UPPERCASE
#
# Example:
#   BEFORE: ['Year', 'Gold', 'Silver', 'Bronze', 'Total']
#   AFTER:  ['YEAR', 'GOLD', 'SILVER', 'BRONZE', 'TOTAL']
#
# USAGE RULES:
#   - ALWAYS use UPPERCASE: df['YEAR'], df['GOLD'], df['SILVER']
#   - NEVER use lowercase: df['Year'], df['gold'] (will cause KeyError)
#   - NEVER guess column names - check with df.columns first
# ============================================================================

# from ... import ...

# The model class
class Model3():
    pass

# The function to complete the current Task
def task3():
    # ...
    # print(result) or save generated image to ./task1.png
    # Note: The output must print the necessary explanations.
    return

if __name__ == '__main__':
    # complete task
    task3()
