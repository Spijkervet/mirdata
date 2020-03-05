# -*- coding: utf-8 -*-
"""Billboard Dataset Loader
"""

import csv
import librosa
import logging
import numpy as np
import os
import shutil
import re

import mirdata.utils as utils
import mirdata.download_utils as download_utils
import mirdata.jams_utils as jams_utils

DATASET_DIR = "McGill-Billboard"
INDEX_REMOTE = download_utils.RemoteFileMetadata(
    filename="billboard-2.0-index.csv",
    url="https://www.dropbox.com/s/o0olz0uwl9z9stb/billboard-2.0-index.csv?dl=1",
    checksum="c47d304c212725998839cf9bb1a417aa",
    destination_dir=None,
)

ANNOTATIONS_REMOTE = download_utils.RemoteFileMetadata(
    filename="billboard-2.0-salami_chords.tar.xz",
    url="https://www.dropbox.com/s/p4xtixbvt4hw5c6/billboard-2.0-salami_chords.tar.xz?dl=1",
    checksum="201b0c3c72cefe6c2fe697dc7886ce0d",
    destination_dir=None,
)

CHORDS_REMOTE = download_utils.RemoteFileMetadata(
    filename="billboard-2.0.1-lab.tar.xz",
    url="https://www.dropbox.com/s/t390alzrkx0c9yt/billboard-2.0.1-lab.tar.xz?dl=1",
    checksum="a7b1fa6a7e454bf73ced7c29207aa597",
    destination_dir=None,
)

CHORDS_MIREX13 = download_utils.RemoteFileMetadata(
    filename="billboard-2.0.1-mirex.tar.xz",
    url="https://www.dropbox.com/s/fg8lvy79o7etiyc/billboard-2.0.1-mirex.tar.xz?dl=1",
    checksum="97e5754699f3b45aa5cc70d8a7611c54",
    destination_dir=None,
)


def _load_metadata(data_home):

    metadata_path = os.path.join(data_home, "billboard-2.0-index.csv")

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    with open(metadata_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        next(reader, None)
        raw_data = []
        for line in reader:
            if line != []:
                raw_data.append(line)

    metadata_index = {}
    for line in raw_data:
        track_id = line[0]
        metadata_index[track_id] = {
            "chart_date": line[1],
            "target_rank": int(line[2]) if line[2] else None,
            "actual_rank": int(line[3]) if line[3] else None,
            "title": line[4],
            "artist": line[5],
            "peak_rank": int(line[6]) if line[6] else None,
            "weeks_on_chart": int(line[7]) if line[7] else None,
        }

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = utils.LargeData("billboard_index.json", _load_metadata)


class Track(object):
    """SALAMI Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        track_id (str): track id
        audio_path (str): audio path of the track
        chart date: release date of the track
        target rank: 
        actual rank: 
        title: title of the track
        artist: artist name
        peak rank: 
        weeks on chart: 
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError("{} is not a valid track ID in Salami".format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata.keys():
            self._track_metadata = metadata[track_id]
        else:
            # annotations with missing metadata
            self._track_metadata = {
                "chart_date": None,
                "target_rank": None,
                "actual_rank": None,
                "title": None,
                "artist": None,
                "peak_rank": None,
                "weeks_on_chart": None,
            }

        self.salami_path = os.path.join(self._data_home, self._track_paths["salami"][0])
        self.lab_full_path = os.path.join(
            self._data_home, self._track_paths["lab_full"][0]
        )
        self.lab_majmin7_path = os.path.join(
            self._data_home, self._track_paths["lab_majmin7"][0]
        )
        self.lab_majmin7inv_path = os.path.join(
            self._data_home, self._track_paths["lab_majmin7inv"][0]
        )
        self.lab_majmin_path = os.path.join(
            self._data_home, self._track_paths["lab_majmin"][0]
        )
        self.lab_majmininv_path = os.path.join(
            self._data_home, self._track_paths["lab_majmininv"][0]
        )

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.chart_date = self._track_metadata["chart_date"]
        self.target_rank = self._track_metadata["target_rank"]
        self.actual_rank = self._track_metadata["actual_rank"]
        self.title = self._track_metadata["title"]
        self.artist = self._track_metadata["artist"]
        self.peak_rank = self._track_metadata["peak_rank"]
        self.weeks_on_chart = self._track_metadata["weeks_on_chart"]

    def __repr__(self):
        repr_string = (
            "Billboard Track(track_id={}, audio_path={}, "
            + "chart_date={}, "
            + "title={}, artist={}, "
            + "SectionData('intervals', 'labels')"
        )
        return repr_string.format(
            self.track_id, self.audio_path, self.chart_date, self.title, self.artist
        )

    @utils.cached_property
    def chords(self):
        return {
            "full": _load_chords(self.lab_full_path),
            "majmin7": _load_chords(self.lab_majmin7_path),
            "majmin7inv": _load_chords(self.lab_majmin7inv_path),
            "majmin": _load_chords(self.lab_majmin_path),
            "majmininv": _load_chords(self.lab_majmininv_path),
        }

    def salami_metadata(self):
        return _parse_salami_metadata(
            os.path.join(self._data_home, self._track_paths["salami"][0])
        )

    @utils.cached_property
    def sections(self):
        return _load_sections(
            os.path.join(self._data_home, self._track_paths["salami"][0])
        )

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=None, mono=True)

    def to_jams(self):
        return jams_utils.jams_converter(
            chord_data=[
                (self.chords["full"], "full"),
                (self.chords["majmin7"], "majmin7"),
                (self.chords["majmin7inv"], "majmin7inv"),
                (self.chords["majmin"], "majmin"),
                (self.chords["majmininv"], "majmininv"),
            ],
            section_data=[(self.sections, None)],
            metadata=self._track_metadata,
        )


def _load_chords(path):
    """Private function to load LAB format chord data from a file

    Args:
        chords_path (str):

    """

    if path is None or not os.path.exists(path):
        return None

    start_times, end_times, chords = [], [], []
    with open(path, "r") as f:
        for l in f:
            l = l.rstrip()
            if l:
                start, end, label = l.split("\t")
                start_times.append(float(start))
                end_times.append(float(end))
                chords.append(label)

    chord_data = utils.ChordData(np.array([start_times, end_times]).T, chords)
    return chord_data


def _load_sections(sections_path):
    """Private function to load SALAMI format sections data from a file

    Args:
        sections_path (str):

    """
    if sections_path is None or not os.path.exists(sections_path):
        return None

    salami = _parse_salami(sections_path)
    timed_sections = _timed_sections(salami)

    # Clean sections
    timed_sections_clean = [ts for ts in timed_sections if ts["section"] is not None]

    start_times, end_times, sections = [], [], []
    for idx, ts in enumerate(timed_sections_clean):
        if idx < len(timed_sections_clean) - 1:
            start_times.append(timed_sections_clean[idx]["time"])
            end_times.append(timed_sections_clean[idx + 1]["time"])
            sections.append(timed_sections_clean[idx]["section"])
        else:
            start_times.append(timed_sections_clean[idx]["time"])
            end_times.append(timed_sections[-1]["time"])  # end of song
            sections.append(timed_sections_clean[idx]["section"])

    section_data = utils.SectionData(np.array([start_times, end_times]).T, sections)

    return section_data


def _parse_salami_metadata(fn):
    s = open(fn).read().split("\n")
    o = {}
    for x in s:
        if x.startswith("#"):
            if x[2:].startswith("title:"):
                o["title"] = x[9:]
            if x[2:].startswith("artist:"):
                o["artist"] = x[10:]
            if x[2:].startswith("metre:"):
                o["meter"] = o.get("meter", []) + [x[9:]]
            if x[2:].startswith("tonic:"):
                o["tonic"] = o.get("tonic", []) + [x[9:]]
        else:
            break
    return o


def _parse_salami(fn):
    """
        Author:
            Brian Whitman
            brian@echonest.com
            https://gist.github.com/bwhitman/11453443

        Parse a salami_chords.txt file and return a dict with all the stuff innit
    """
    s = open(fn).read().split("\n")
    o = {}
    for x in s:
        if x.startswith("#"):
            if x[2:].startswith("title:"):
                o["title"] = x[9:]
            if x[2:].startswith("artist:"):
                o["artist"] = x[10:]
            if x[2:].startswith("metre:"):
                o["meter"] = o.get("meter", []) + [x[9:]]
            if x[2:].startswith("tonic:"):
                o["tonic"] = o.get("tonic", []) + [x[9:]]
        elif len(x) > 1:
            spot = x.find("\t")
            if spot > 0:
                time = float(x[0:spot])
                event = {}
                event["time"] = time
                rest = x[spot + 1 :]
                items = rest.split(", ")
                for i in items:
                    chords = re.findall(r"(?=\| (.*?) \|)", i)
                    section = i.split("|")
                    if len(section) == 1 and not ("(" in section or ")" in section):
                        event["section"] = section[0]
                    if len(chords):
                        event["chords"] = chords
                    else:
                        event["notes"] = event.get("notes", []) + [i]
                o["events"] = o.get("events", []) + [event]
    return o


def _timed_sections(parsed):
    """
        Author:
            Brian Whitman
            brian@echonest.com
            https://gist.github.com/bwhitman/11453443

        Given a salami parse return a list of parsed chords with timestamps & deltas
    """
    timed_sections = []
    tic = 0
    for i, e in enumerate(parsed["events"]):
        sections = []
        try:
            dt = parsed["events"][i + 1]["time"] - e["time"]
        except IndexError:
            dt = 0

        section = None
        if e.get("notes"):
            if len(e.get("notes")) > 1:
                section = (e.get("notes")[0], e.get("notes")[1])
            sections.append(section)

        tic = e["time"]
        if len(sections):
            seconds_per_chord = dt / float(len(sections))
            for c in sections:
                timed_sections.append(
                    {"time": tic, "section": c, "length": seconds_per_chord}
                )
                tic = tic + seconds_per_chord
    return timed_sections


def download(data_home=None, force_overwrite=False):
    """Download Billboard Dataset (annotations).
    The audio files are not provided.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

        force_overwrite (bool): whether to overwrite the existing downloaded data

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    info_message = """
        Unfortunately the audio files of the McGill-Billboard dataset are not available
        for download. If you have the McGill-Billboard dataset, place the contents into a
        folder called McGill-Billboard with the following structure:
            > McGill-Billboard/
                > annotations/ 
                > audio/
        and copy the McGill-Billboard folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        tar_downloads=[ANNOTATIONS_REMOTE, CHORDS_REMOTE, CHORDS_MIREX13],
        file_downloads=[INDEX_REMOTE],
        info_message=info_message,
        force_overwrite=force_overwrite,
    )

    annotations_dir = os.path.join(data_home, "annotations")
    if force_overwrite:
        if os.path.exists(annotations_dir):
            shutil.rmtree(annotations_dir)
        shutil.move(os.path.join(data_home, DATASET_DIR), annotations_dir)
    else:
        if os.path.exists(os.path.join(data_home, DATASET_DIR)):
            if not os.path.exists(annotations_dir):
                shutil.move(os.path.join(data_home, DATASET_DIR), annotations_dir)
            else:
                shutil.rmtree(os.path.join(data_home, DATASET_DIR))


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load Billboard dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in track_ids():
        data[key] = Track(key, data_home=data_home)
    return data


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
John Ashley Burgoyne, Jonathan Wild, and Ichiro Fujinaga, ‘An Expert Ground Truth Set for Audio Chord Recognition and Music Analysis’, in Proceedings of the 12th International Society for Music Information Retrieval Conference, ed. Anssi Klapuri and Colby Leider (Miami, FL, 2011), pp. 633–38 [1]
John Ashley Burgoyne, ‘Stochastic Processes and Database-Driven Musicology’ (PhD diss., McGill University, Montréal, Québec, 2012) [2].

========== Bibtex ==========
@inproceedings{burgoyne2011expert,
  title={An Expert Ground Truth Set for Audio Chord Recognition and Music Analysis.},
  author={Burgoyne, John Ashley and Wild, Jonathan and Fujinaga, Ichiro},
  booktitle={ISMIR},
  volume={11},
  pages={633--638},
  year={2011}
}
"""

    print(cite_data)
