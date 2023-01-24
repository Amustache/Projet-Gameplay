import re


import matplotlib.pyplot as plt
import pandas as pd


FILE = "inputs/2022-11-23-12_13_44.csv"
MODEL = ""


def get_df_trim():
    df_trim = extract_keys_from_file(FILE)
    return df_trim


def get_data():
    df_trim = get_df_trim()
    data = extract_timeline_from_df(df_trim)
    return data


def extract_keys_from_file(file):
    df = pd.read_csv(file)

    # Only keep keys that are of our interest
    df_trim = df[(df["KEY"] == "Key.left") | (df["KEY"] == "Key.right") | (df["KEY"] == "x")]

    df_trim.loc[(df_trim["KEY"] == "Key.right") & (df_trim["STATUS"] == "DOWN"), "action"] = "R"
    df_trim.loc[(df_trim["KEY"] == "Key.right") & (df_trim["STATUS"] == "UP"), "action"] = "r"

    df_trim.loc[(df_trim["KEY"] == "Key.left") & (df_trim["STATUS"] == "DOWN"), "action"] = "L"
    df_trim.loc[(df_trim["KEY"] == "Key.left") & (df_trim["STATUS"] == "UP"), "action"] = "l"

    df_trim.loc[(df_trim["KEY"] == "x") & (df_trim["STATUS"] == "DOWN"), "action"] = "J"
    df_trim.loc[(df_trim["KEY"] == "x") & (df_trim["STATUS"] == "UP"), "action"] = "j"

    df_trim = df_trim[["FRAME", "action"]].reset_index(drop=True)

    return df_trim


# Extract timeline
def extract_timeline_from_df(df_trim):
    start = df_trim.FRAME.min()

    # r, j, l
    data = {"R": [], "L": [], "J": []}
    r = 0
    l = 0
    j = 0
    for index, row in df_trim.iterrows():
        if row.action == "R".upper():
            data["R"].append([row.FRAME])
        if row.action == "R".lower():
            data["R"][-1].append(row.FRAME - data["R"][-1][0])

        if row.action == "L".upper():
            data["L"].append([row.FRAME])
        if row.action == "L".lower():
            data["L"][-1].append(row.FRAME - data["L"][-1][0])

        if row.action == "J".upper():
            data["J"].append([row.FRAME])
        if row.action == "J".lower():
            data["J"][-1].append(row.FRAME - data["J"][-1][0])

    return data


# Generate "barcode"
def fig_generate_barcode(data):
    fig, ax = plt.subplots()
    ax.broken_barh(data["R"], (10, 9), facecolors="tab:red", label="Right")
    ax.broken_barh(data["L"], (20, 9), facecolors="tab:green", label="Left")
    ax.broken_barh(data["J"], (30, 9), facecolors="tab:blue", label="Jump")
    plt.legend(loc="upper right")
    return fig


# Extract patterns
def extract_pattern_from_df(df_trim):
    all_keys = "".join(list(df_trim["action"]))
    return all_keys


def getAllSubStrings(x, l=None, freq=False):
    if not l:
        l = len(x)

    allSubStrings = [x[i : i + l] for i in range(0, len(x)) if len(x[i : i + l]) == l]

    if freq:
        return allSubStrings
    else:
        return set(allSubStrings)


def extract_freq(all_keys, l=3, sort="value"):
    res = {}
    substrings = getAllSubStrings(all_keys, l=l, freq=True)
    for idx in substrings:
        if idx not in res.keys():
            res[idx] = 1
        else:
            res[idx] += 1
    if sort == "key":
        return dict(sorted(res.items(), key=lambda item: item[0], reverse=False))
    elif sort == "value":
        return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))


def find_all_patterns(all_keys):
    all_patterns = {}
    for i in range(4, 11):
        all_patterns.update(extract_freq(all_keys, l=i, sort="value"))
    all_patterns = dict(sorted(all_patterns.items(), key=lambda item: item[1], reverse=True))
    return all_patterns


# First 10 patterns
def fig_10_patterns(all_patterns):
    from itertools import islice

    n_items = list(islice(all_patterns.items(), 10))
    # print(n_items)
    keys = []
    values = []
    for i, (k, v) in enumerate(n_items):
        k = k.replace("J", "J(")
        k = k.replace("j", ")")
        k = k.replace("R", "R(")
        k = k.replace("r", ")")
        k = k.replace("L", "L(")
        k = k.replace("l", ")")
        keys.append(k)
        values.append(v)

    fig, ax = plt.subplots()
    ax.bar(keys, values)
    return fig


# Where are the patterns
def place_patterns(all_patterns):
    place_patterns = {}
    for pattern in all_patterns.keys():
        place_patterns[pattern] = []
        for m in re.finditer(f"(?={pattern})", all_keys):
            place_patterns[pattern].append(df_trim.iloc[m.start()].FRAME)
    return place_patterns


def fig_find_pattern(data, all_patterns, pattern="JjJj"):
    placed_patterns = place_patterns(all_patterns)
    fig, ax = plt.subplots()
    ax.broken_barh(data["R"], (10, 9), facecolors="tab:red")
    ax.broken_barh(data["L"], (20, 9), facecolors="tab:green")
    ax.broken_barh(data["J"], (30, 9), facecolors="tab:blue")
    ax.scatter(placed_patterns["JjJj"], len(placed_patterns["JjJj"]) * [40], c="red")
    return fig
