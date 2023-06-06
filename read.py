from pathlib import Path
import re
import numpy as np
import pandas as pd


def extract_info_from_log_evaluation(file_path, arguments):

    if not file_path.exists():
        return {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        info = {}

        for arg in arguments:
            pattern = r'\|\s+{}:\s+([^\n]+)'.format(arg)
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    value = match.group(1).strip()
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                    info[arg] = value
                    break
            else:
                info[arg] = None

        # Extract Loglik (V) value and error
        loglik_value = None
        loglik_error = None
        for i in range(len(lines)):
            if "Interpolation in training range" in lines[i]:
                loglik_line = lines[i + 1]  # Next line after "Interpolation in training range:"
                match = re.match(r"^\d{2}:\d{2}:\d{2} \| +Loglik \(V\): +([-+]?[0-9]*\.?[0-9]+) \+- +([0-9]*\.?[0-9]+)", loglik_line)
                if match:
                    loglik_value = float(match.group(1))
                    loglik_error = float(match.group(2))
                break

        info['loglik_mean'] = loglik_value
        info['loglik_error'] = loglik_error

        return info


def extract_epoch_duration(file_path):
    epoch_times = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'\d{2}:\d{2}:\d{2}\s\|\sEpoch\s\d+', line)
            if match:
                time_str = line.split('|')[0].strip()
                hours, minutes, seconds = map(int, time_str.split(':'))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                epoch_times.append(total_seconds)
    
    if len(epoch_times) >= 2:
        epoch_times = np.array(epoch_times)
        epoch_duration = epoch_times[1:] - epoch_times[:-1]
        return np.mean(epoch_duration)
    else:
        None


def find_last_epoch(file_path):
    largest_epoch = -1
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            match = re.search(r'Epoch (\d+)', line)
            if match:
                largest_epoch = int(match.group(1))
                break
    return largest_epoch


def get_np_data(root_dir, target_filename):
    data = []
    root_path = Path(root_dir)
    for file_path in root_path.glob('**/' + target_filename):
        relative_path = file_path.relative_to(root_path)
        folders = relative_path.parent.parts
        experiment_info = {
            "data": folders[0],
            "dim_x": int(folders[1].split("_")[0][-1]),
            "model": folders[2],
        }
        largest_epoch = find_last_epoch(file_path)
        epoch_duration = extract_epoch_duration(file_path)
        eval_file = file_path.parent / "log_evaluate.txt"
        if eval_file.exists():
            eval_info = extract_info_from_log_evaluation(eval_file, list(experiment_info.keys()))
            # check file path info and eval info are the same
            for k, v in experiment_info.items():
                assert eval_info[k] == v, f"{eval_info[k]} != {v}"
            
            experiment_data = eval_info
        else:
            experiment_data = experiment_info

        data.append({
            "max_epoch": largest_epoch,
            "mean_epoch_duration": epoch_duration,
            **experiment_data
        })
            
    return data

# Usage example
root_directory = '_experiments'  # Replace with the actual root directory path
target_file = 'log_train.txt'  # Replace with the specific filename
data_np = get_np_data(root_directory, target_file)


# Models from 'regression-May25-2'
# See regression-May29-eval for the evaluation results
# num_layers = 4
# num_hidden = 64
# float64 = False
data_ndp = [
    # EQ
    {"model": "ndp", "data": "eq", "dim_x": 1, "loglik_mean": 0.47832477, "loglik_error": 0.03662583},
    {"model": "ndp", "data": "eq", "dim_x": 2, "loglik_mean": -0.67219281, "loglik_error": 0.04626374},
    {"model": "ndp", "data": "eq", "dim_x": 3, "loglik_mean": -1.15707684, "loglik_error": 0.03844018},
    # Matern: TODO
    {"model": "ndp", "data": "matern", "dim_x": 1, "loglik_mean": -0.00083868, "loglik_error": 0.04221253},
    {"model": "ndp", "data": "matern", "dim_x": 2, "loglik_mean": -1.05238593, "loglik_error": 0.03250215},
    {"model": "ndp", "data": "matern", "dim_x": 3, "loglik_mean": -1.32986808, "loglik_error": 0.03265943},
]


data_gp = [
    {'model': 'gp', 'data': 'eq', 'dim_x': 1, 'loglik_mean': 0.667953953271597, 'loglik_error': 0.0284062675864906},
    {'model': 'gp', 'data': 'eq', 'dim_x': 2, 'loglik_mean': -0.4455377081957982, 'loglik_error': 0.03404991121819787},
    {'model': 'gp', 'data': 'eq', 'dim_x': 3, 'loglik_mean': -0.9361569976140109, 'loglik_error': 0.027897607489126185},
    {'model': 'gp', 'data': 'matern', 'dim_x': 1, 'loglik_mean': 0.18864097804478128, 'loglik_error': 0.033781528944967554},
    {'model': 'gp', 'data': 'matern', 'dim_x': 2, 'loglik_mean': -0.8498409649606518, 'loglik_error': 0.026913324180852388},
    {'model': 'gp', 'data': 'matern', 'dim_x': 3, 'loglik_mean': -1.1387292239297335, 'loglik_error': 0.020928975026162592},
    {'model': 'gp_diag', 'data': 'eq', 'dim_x': 1, 'loglik_mean': -0.8793676809905253, 'loglik_error': 0.07339923818410686},
    {'model': 'gp_diag', 'data': 'eq', 'dim_x': 2, 'loglik_mean': -1.0362116659485214, 'loglik_error': 0.04409327729855993},
    {'model': 'gp_diag', 'data': 'eq', 'dim_x': 3, 'loglik_mean': -1.2184787611104158, 'loglik_error': 0.03598832323616035},
    {'model': 'gp_diag', 'data': 'matern', 'dim_x': 1, 'loglik_mean': -0.9786567394013844, 'loglik_error': 0.06004747045716921},
    {'model': 'gp_diag', 'data': 'matern', 'dim_x': 2, 'loglik_mean': -1.2148931971423933, 'loglik_error': 0.040306563965239815},
    {'model': 'gp_diag', 'data': 'matern', 'dim_x': 3, 'loglik_mean': -1.3125737233647539, 'loglik_error': 0.02795245130271025},
]


def format_number(value, error, *, bold=False, possibly_negative=True):
    if np.isnan(value):
        return ""
    if np.abs(value) > 10 or np.abs(error) > 10:
        return "F"
    if value >= 0 and possibly_negative:
        sign_spacer = "\\hphantom{-}"
    else:
        sign_spacer = ""
    if bold:
        bold_start, bold_end = "\\mathbf{", "}"
    else:
        bold_start, bold_end = "", ""
    return f"${sign_spacer}{bold_start}{value:.2f}{bold_end} {{ \\pm \\scriptstyle {error:.2f} }}$"



def n_best(df, col, col_err, *, ascending):
    df = df.sort_values(col, ascending=ascending)
    best_indices = set()
    val0, err0 = df[[col, col_err]].iloc[0]
    i = 0
    while True and i < len(df) - 1:
        best_indices.add(df.index[i])

        # Compare with the next.
        val, err = df[[col, col_err]].iloc[i + 1]
        diff = abs(val0 - val)
        diff_err = np.sqrt(err0**2 + err**2)

        if diff > diff_err:
            # Significantly better.
            return best_indices
        else:
            # Not significantly better. Try the next.
            i += 1

    return {}

metrics = ["loglik_mean", "loglik_error"]


df = pd.DataFrame(data_np + data_gp + data_ndp)
df2 = df.groupby(["model", "data", "dim_x"]).agg({ m: ["first"] for m in metrics})
df2 = df2.unstack([1,2])  # makes data and dim_x into columns
df2 = df2.droplevel(1, axis=1)  # removes the "first" level
# sort by rank
rank = df2["loglik_mean"].rank().mean(axis=1, skipna=True)
df2["rank"] = rank
df2 = df2.sort_values("rank", ascending=False)


for col in df2["loglik_mean"].columns:
    col_value = ("loglik_mean", *col)
    col_err = ("loglik_error", *col)
    indices = n_best(df2[~df2.index.isin(["gp", "gp_diag"])], col_value, col_err, ascending=False)
    # set best to True and other to false
    df2.loc[:, ("best", *col)] = False
    df2.loc[list(indices), ("best", *col)] = True


def format_row(row):
    for col in df2["loglik_mean"].columns:
        v = row[("loglik_mean", *col)]
        e = row[("loglik_error", *col)]
        row[("table", *col)] = format_number(v, e, bold=row[("best", *col)], possibly_negative=True)

    return row
    

RENAME = {
    "gp": "GP {(truth)}",
    "gp_diag": "GP {(diag)}",
    "ndp": "NDP",
    "eq": "Squared Exponential",
    "matern": r"Matérn-$\frac52$",
    "convcnp": "ConvCNP",
    "convnp": "ConvNP",
}

def display_name(name):
    return RENAME.get(name, name.upper())

df_table = df2.apply(format_row, axis=1)["table"]
df_table.index = [f"\\scshape {display_name(x)}" for x in df_table.index]
df_table.columns = pd.MultiIndex.from_tuples(
    [(display_name(dataset), f"$D = {dim}$") for dataset, dim in df_table.columns]
)
print(
    df_table.style.to_latex(column_format="l" + "c" * 6, hrules=True, multicol_align="c")
)
    

