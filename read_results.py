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
        for dim_x in [1, 2]:
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
    return f"${sign_spacer}{bold_start}{value:.2f}{bold_end} {{ \\pm \\small {error:.2f} }}$"


def format_table(title1, title2, df, cols, *, possibly_negative, ascending=True):
    for col in cols:
        col["best"] = n_best(df, col["value"], col["error"], ascending=ascending)

    res = f"\\begin{{tabular}}[t]{{l{'c' * len(cols)}}} \n"
    res += "\\toprule \n"
    res += title1
    for col in cols:
        if title2:
            res += " & \\multirow{2}{*}{" + col["name"] + "}"
        else:
            res += " & " + col["name"]
    if title2:
        res += " \\\\ \n"
        res += title2 + " \\\\ \\midrule \n"
    else:
        res += " \\\\ \\midrule \n"
    for name, row in df.iterrows():
        res += name
        for col in cols:
            res += " & " + format_number(
                row[col["value"]],
                row[col["error"]],
                bold=name in col["best"],
                possibly_negative=possibly_negative,
            )
        res += " \\\\ \n"
    res += "\\bottomrule \\\\ \n"
    res += "\\end{tabular}"
    return res


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
df = df.set_index(["data", "dim_x", "name"])
df = pd.pivot_table(df, index=['name'], columns=['data', 'dim_x'], fill_value=np.nan)

print(df)

DATASETS = ["eq", "weakly-periodic", "matern"]
XDIMS = [1, 2]
COLS = list(it.product(DATASETS, XDIMS))


table = f"\\begin{{tabular}}{{l{'c' * len(COLS)}}} \n"
table += "\\toprule \n"

header1 = '& '.join([f"\\multicolumn{{{len(XDIMS)}}}{{c}}{{{d}}}" for d in DATASETS])
table += '&' + header1 + '\\\\ \n'

header2 = '& '.join([f"Dx = {xdim}" for _ in DATASETS for xdim in XDIMS])
table += '&' + header2 + ' \\\\ \n'

table += " \\midrule \n"


for name, data in df.iterrows():
    table += name
    for dataset, xdim in it.product(DATASETS, XDIMS):
        val = data.xs(dataset, level="data").xs(xdim, level="dim_x")["int"]
        err = data.xs(dataset, level="data").xs(xdim, level="dim_x")["int-err"]
        table += " & " + format_number(val, err, bold=False, possibly_negative=True)
    table += " \\\\ \n"

table += "\\bottomrule \n"
table += "\\end{tabular}"
print(table)

# columns = [
#     {"name": "Interp.", "value": "int", "error": "int-err"},
# ]

# ascending = False
# possible_negative = True

# datasets = [
#     {"name": "EQ", "key": "eq"},
#     {"name": "WeaklyPeriodic", "key": "weakly-periodic"},
# ]
# xdims = [1, 2]


# res = ""

# for row in rows:
#     res += row["name"]
#     for dataset in datasets:
#         for xdim in xdims:
#             df_ = (df
#                 .xs(row["name"], level="name")
#                 .xs(dataset["key"], level="data")
#                 .xs("loglik", level="kind")
#                 .xs(xdim, level="dim_x")
#                 .xs(1, level="dim_y")
#                 .sort_values("int", ascending=ascending)
#             )





# title1 = "EQ"
# title2 = "Dx = 1"
# table = format_table(
#     title1,
#     title2,
#     df.xs("eq", level="data").xs("loglik", level="kind").xs(1, level="dim_x").xs(1, level="dim_y").sort_values("int", ascending=ascending),
#     columns,
#     possibly_negative=True,
#     ascending=ascending,
# )
# print(table)