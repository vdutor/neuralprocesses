import subprocess
import itertools as it

import numpy as np
import pandas as pd
import wbml.parser as parser


def parse_logs():
    logs = (
        subprocess.check_output(
            f"find _experiments | grep log_evaluate_out.txt",
            shell=True,
        )
        .strip()
        .splitlines(keepends=False)
    )
    results = {}
    for log in logs:
        print("reading log:", log)
        run = log.decode().split("/")[1:-1]
        for header in ["loglik"]:
            p = parser.Parser(log)
            try:
                p.find_line(header + ":")
            except RuntimeError:
                continue
            for task, desc in [
                ("int", "interpolation in training range"),
                # ("int-beyond", "interpolation beyond training range"),
                # ("extr", "extrapolation beyond training range"),
            ]:
                p.find_line(desc)
                p.next_line()
                res = p.parse(
                    parser.SkipUntil("|"),
                    parser.Whitespace(),
                    parser.Literal("Loglik (V):"),
                    parser.Whitespace(),
                    parser.Float(),
                    parser.Whitespace(),
                    parser.Literal("+-"),
                    parser.Whitespace(),
                    parser.Float(),
                )
                results[tuple(run) + (header, task, "loglik")] = res
    return results


# Build data frame.
def build_entries(results, rows):
    entries = []
    for data in ["eq", "matern", "weakly-periodic", "sawtooth", "mixture"]:
        for dim_x in [1, 2, 3]:
            for dim_y in [1]:
                for row in rows:
                    # Build the entry for kind `loglik`.
                    entry = {
                        "kind": "loglik",
                        "data": data,
                        "dim_x": dim_x,
                        "dim_y": dim_y,
                        "name": row["name"],
                    }
                    for task in ["int"]:
                        try:
                            val, err = results[
                                (data, f"x{dim_x}_y{dim_y}", *row["key"], task, "loglik")
                            ]
                            entry[task] = val
                            entry[task + "-err"] = err
                        except KeyError:
                            continue
                    entries.append(entry)
    return entries


def n_best(df, col, col_err, *, ascending):
    df = df.sort_values(col, ascending=ascending)
    best_indices = set()
    val0, err0 = df[[col, col_err]].iloc[0]
    i = 0
    while True:
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



results = parse_logs()

rows = [
    {"name": "CNP", "key": ("cnp", "loglik", "loglik")},
    {"name": "NP", "key": ("np", "loglik", "loglik")},
    {"name": "ANP", "key": ("anp", "loglik", "loglik")},
]
entries = build_entries(results, rows=rows)

df = pd.DataFrame(entries)
df = df[df.data.isin(["eq", "matern", "weakly-periodic"])]
del df["kind"]
del df["dim_y"]

DATAFRAME_NDP = [
    {"name": "NDP", "data": "eq", "dim_x": 1, "int": 0.38, "int-err": 0.05},
    {"name": "trivial", "data": "eq", "dim_x": 1, "int": -1.41, "int-err": 0.03},
    {"name": "NDP", "data": "eq", "dim_x": 2, "int": -1.01, "int-err": 0.03},
    {"name": "trivial", "data": "eq", "dim_x": 2, "int": -1.42, "int-err": 0.02},
    {"name": "NDP", "data": "matern", "dim_x": 1, "int": -0.13, "int-err": 0.05},
    {"name": "trivial", "data": "matern", "dim_x": 1, "int": -0.43, "int-err": 0.02},
    {"name": "NDP", "data": "matern", "dim_x": 2, "int": -1.15, "int-err": 0.02},
    {"name": "trivial", "data": "matern", "dim_x": 2, "int": -1.43, "int-err": 0.02},
]
df = pd.concat([df, pd.DataFrame(DATAFRAME_NDP)])

df = df.set_index(["data", "dim_x", "name"])
df = pd.pivot_table(df, index=['name'], columns=['data', 'dim_x'], fill_value=np.nan)

df["int"] = -df["int"]
possible_negative = True
ascending = True
show_rank = False

rank = df.rank().mean(axis=1)
df = df.iloc[np.argsort(rank.values)]

DATASETS = ["eq", "matern"]
DATASET_NAME = {
    "eq": "Squared Exponential",
    "matern": r"Mat\'ern",
}
# DATASETS = ["eq", "matern"]
XDIMS = [1, 2, 3]
COLS = [
    {"dataset": dataset, "xdim": xdim}
    for dataset, xdim in list(it.product(DATASETS, XDIMS))
]


table = f"\\begin{{tabular}}{{l{'c' * len(COLS)}{'c' if show_rank else ''}}} \n"
table += "\\toprule \n"

header1 = '& '.join([f"\\multicolumn{{{len(XDIMS)}}}{{c}}{{{DATASET_NAME[d]}}}" for d in DATASETS])
if show_rank:
    header1 += r'& \multirow{2}{*}{av. rank}'
table += '&' + header1 + '\\\\ \n'


header2 = '& '.join([f"$D_x = {xdim}$" for _ in DATASETS for xdim in XDIMS])
table += '&' + header2 + ' \\\\ \n'

table += " \\midrule \n"

for col in COLS:
    col["best"] = n_best(
        df.xs(col["dataset"], level="data", axis=1).xs(col["xdim"], level="dim_x", axis=1),
        "int",
        "int-err",
        ascending=ascending
    )

for name, data in df.iterrows():
    table += name
    for col in COLS:
        dataset, xdim, best = col["dataset"], col["xdim"], col["best"]
        val = data.xs(dataset, level="data").xs(xdim, level="dim_x")["int"]
        err = data.xs(dataset, level="data").xs(xdim, level="dim_x")["int-err"]
        table += " & " + format_number(val, err, bold=name in best, possibly_negative=possible_negative)
    if show_rank:
        table += f' & {rank[name]:.2f}'
    table += " \\\\ \n"

table += "\\bottomrule \n"
table += "\\end{tabular}"
print(table)
