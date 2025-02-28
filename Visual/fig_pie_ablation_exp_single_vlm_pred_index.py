import os

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = 'Result/Individual VLM Experiment phi newbenchmark/1740680205/result.csv'
data = pd.read_csv(csv_file)

# Extract the 'pred option' column
pred_option = data['pred option']

# Count the occurrences of each unique value in the 'pred option' column
pred_option_counts = pred_option.value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(pred_option_counts, labels=pred_option_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Prediction Options Distribution')

# Save the pie chart as a PDF file
os.makedirs('./Visual/fig', exist_ok=True)
output_file = './Visual/fig/pie_demo.pdf'
plt.savefig(output_file)

# Show the plot (optional)
# plt.show()