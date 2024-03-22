import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'const_vel_path_data.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Select and rename the columns 'pos_X' and 'pos_Y' to 'X' and 'Y'
df_renamed = df[['pos_X', 'pos_Y']].rename(columns={'pos_X': 'X', 'pos_Y': 'Y'})

# Scale the 'X' and 'Y' columns by 1/55
df_renamed['X'] = df_renamed['X'] * (1/1)
df_renamed['Y'] = df_renamed['Y'] * (1/1)

# Save the new DataFrame to a new CSV file
new_csv_path = 'const_vel_path_data_new.csv'
df_renamed.to_csv(new_csv_path, index=False)

