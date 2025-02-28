import os
import json

result_json_path = 'Result/Pair VLM Exp on phi newbenchmark/1740621448/result.json'

with open(result_json_path, 'r') as f:
    result_json = json.load(f)

# Assuming each_result is a dictionary and you want to include specific keys in the table
keys_to_include = ['reason', 'question']

latex_table = "\\begin{table}[tb]\n\\centering\n\\begin{tabular}{" + "p{5cm}" * (len(keys_to_include)) + "}\n\\toprule\n"
latex_table += " & ".join(f"\\multicolumn{{1}}{{c}}{{\\textbf{{{key.upper()}}}}}" for key in keys_to_include) + " \\\\\n\\midrule\n"

for i in range(20):
    each_result = result_json[i]
    keys_to_include_temp = [key for key in keys_to_include if key != 'error type']
    row = [str(each_result[key]) for key in keys_to_include_temp]
    latex_table += " & ".join(row) + "\\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{Your Caption Here}\n\\label{table:your_label}\n\\end{table}"

# Save the LaTeX table to a .txt file
latex_output_dir = 'Visual/table/'
os.makedirs(latex_output_dir, exist_ok=True)
latex_output_path = os.path.join(latex_output_dir, 'latex_table_error_analysis_drop_primary_keys.txt')
with open(latex_output_path, 'w') as f:
    f.write(latex_table)

# New table with 'scene', 'seq', 'pair'
keys_to_include_new = ['scene', 'seq', 'pair', 'label text', 'round idx', 'pred text', "error type"]

latex_table_new = "\\begin{table}[tb]\n\\centering\n\\begin{tabular}{" + "l" * len(keys_to_include_new) + "}\n\\toprule\n"
latex_table_new += " & ".join(f"\\multicolumn{{1}}{{c}}{{\\textbf{{{'RND' if key == 'round idx' else 'LABEL' if key == 'label text' else 'PRED' if key == 'pred text' else key.upper()}}}}}" for key in keys_to_include_new) + " \\\\\n\\midrule\n"

for i in range(20):
    each_result = result_json[i]
    keys_to_include_temp = [key for key in keys_to_include_new if key != 'error type']
    row = []
    for key in keys_to_include_temp:
        value = str(each_result[key])
        if key == 'seq':
            value = value.replace('seq-', '')
        elif key == 'pair':
            value = '-'.join([part.lstrip('0') if len(part.lstrip('0')) > 2 else part[-3:] for part in value.split('-')])
        row.append(value)
    latex_table_new += " & ".join(row) + " \\\\\n"

latex_table_new += "\\bottomrule\n\\end{tabular}\n\\caption{Your Caption Here}\n\\label{table:your_label_new}\n\\end{table}"

# Save the new LaTeX table to a .txt file
latex_output_path_new = os.path.join(latex_output_dir, 'latex_table_error_analysis_primary_keys.txt')
with open(latex_output_path_new, 'w') as f:
    f.write(latex_table_new)
