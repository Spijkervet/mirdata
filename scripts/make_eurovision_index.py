import argparse
import hashlib
import json
import os
import csv
import unicodedata
import numpy as np

EUROVISION_INDEX_PATH = "../mirdata/indexes/eurovision_index.json"


def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(
                distance[row - 1][col] + 1,  # Cost of deletions
                distance[row][col - 1] + 1,  # Cost of insertions
                distance[row - 1][col - 1] + cost,
            )  # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_audio_idx(audio_dir):
    cds = os.listdir(audio_dir)
    mp3files = []
    for c in sorted(cds):
        for t in sorted(os.listdir(os.path.join(audio_dir, c))):
            if "mp3" in t:
                year = int(c[0:4])
                country = ""
                if "(" in t and ")" in t:
                    country = (
                        os.path.splitext(t.split()[-1])[0]
                        .replace("(", "")
                        .replace(")", "")
                    )

                fp = os.path.join("audio", c, t)
                t = os.path.splitext(" ".join(t.split()[1:]))[0]
                mp3files.append((fp, t, year, country))
    return mp3files


def find_audio(audio_idx, year, country, title):
    for idx_fp, idx_t, idx_year, idx_country in audio_idx:
        if year == idx_year:
            if len(title) < 2:
                continue
            ratio = levenshtein_ratio_and_distance(title, idx_t, ratio_calc=True)
            if ratio >= 0.5:
                return idx_fp
            elif (
                title.lower() in idx_t.lower() or country.lower() in idx_country.lower()
            ):
                return idx_fp
    return "" 


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def make_eurovision_index(data_path):
    annotations_dir = os.path.join(data_path, "Eurovision", "annotations")
    contestants = os.path.join(annotations_dir, "contestants.csv")

    track_id = 0
    track_ids = []
    eurovision_index = {}
    audio_dir = os.path.join(data_path, "Eurovision", "audio")
    audio_idx = make_audio_idx(audio_dir)
    with open(contestants, "r") as f:
        rows = csv.reader(f)
        next(rows)  # skip header
        for row in rows:
            year = int(row[0])
            country = row[2]

            # custom preprocessing:
            if "United Kingdom" in country:
                country.replace("UK", "")

            if "North MacedoniaNorth MacedoniaN.Macedonia" in country:
                country = "North Macedonia"

            title = row[4]
            title = remove_accents(title)

            audio_fp = find_audio(audio_idx, year, country, title)

            if audio_fp:
                # checksum
                audio_checksum = md5(os.path.join(data_path, "Eurovision", audio_fp))

            eurovision_index[track_id] = {
                "audio": (audio_fp, audio_checksum),
            }
            track_ids.append(track_id)
            track_id += 1

    with open(EUROVISION_INDEX_PATH, "w") as fhandle:
        json.dump(eurovision_index, fhandle, indent=2)


def main(args):
    make_eurovision_index(args.eurovision_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Eurovision index file.")
    PARSER.add_argument(
        "eurovision_data_path", type=str, help="Path to Eurovision data folder."
    )

    main(PARSER.parse_args())
