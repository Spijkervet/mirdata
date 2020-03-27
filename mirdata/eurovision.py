# -*- coding: utf-8 -*-
"""Eurovision Dataset Loader

The Eurovision Song Contest is a freely-available dataset containing metadata, contest ranking
and voting data of 1562 songs that have competed in the Eurovision Song Contests.

Every year, the dataset is updated with the contest's results. This release contains
the contestant metadata, contest ranking and voting data of 1562 entries that participated
in the Eurovision Song Contest from its first occurrence in 1956 until now.
The metadata and voting data are provided by the EurovisionWorld fansite.


Attributes:
    DATASET_DIR (str): The directory name for Eurovision dataset. Set to `'Eurovision'`.

    DATA.index (dict): {track_id: track_data}.
        track_data is a jason data loaded from `index/`

    ANNOTATIONS_REMOTE (RemoteFileMetadata (namedtuple)): metadata
        of Eurovision dataset. It includes the annotation file name, annotation
        file url, and checksum of the file.

"""
import csv
import librosa
import os
import logging
import numpy as np

import mirdata.utils as utils
import mirdata.download_utils as download_utils

DATASET_DIR = "Eurovision"
CONTESTANTS_REMOTE = download_utils.RemoteFileMetadata(
    filename="contestants.csv",
    url="https://github.com/Spijkervet/eurovision-dataset/releases/download/1.0/contestants.csv",
    checksum="1c962c90afd957b29ecbe88a74240212",
    destination_dir="annotations",
)

VOTES_REMOTE = download_utils.RemoteFileMetadata(
    filename="votes.csv",
    url="https://github.com/Spijkervet/eurovision-dataset/releases/download/1.0/votes.csv",
    checksum="66c6c4494a27b0f9d2b8febd03658ea7",
    destination_dir="annotations",
)



def _load_metadata(data_home):
    print(data_home)
    metadata_path = os.path.join(data_home, "annotations", "contestants.csv")

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    with open(metadata_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        next(reader, None)
        raw_data = []
        for line in reader:
            raw_data.append(line)

    metadata_index = {}
    track_id = 0
    for line in raw_data:
        (
            year,
            to_country_id,
            to_country,
            performer,
            song,
            sf_num,
            running_final,
            running_sf,
            place_final,
            points_final,
            place_sf,
            points_sf,
            points_tele_final,
            points_jury_final,
            points_tele_sf,
            points_jury_sf,
            composers,
            lyricists,
            lyrics,
            youtube_url,
        ) = line

        metadata_index[str(track_id)] = {
            "year": year,
            "to_country_id": to_country_id,
            "to_country": to_country,
            "performer": performer,
            "song": song,
            "sf_num": sf_num,
            "running_final": running_final,
            "running_sf": running_sf,
            "place_final": place_final,
            "points_final": points_final,
            "place_sf": place_sf,
            "points_sf": points_sf,
            "points_tele_final": points_tele_final,
            "points_jury_final": points_jury_final,
            "points_tele_sf": points_tele_sf,
            "points_jury_sf": points_jury_sf,
            "composers": composers,
            "lyricists": lyricists,
            "lyrics": lyrics,
            "youtube_url": youtube_url,
        }
        track_id += 1

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = utils.LargeData("eurovision_index.json", _load_metadata)


class Track(object):
    """Eurovision track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): track audio path
        title (str): title of the track
        track_id (str): track id

    Properties:
        audio: audio signal, sample rate

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                "{} is not a valid track ID in Eurovision".format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        # self.title = os.path.basename(self._track_paths['sections'][0]).split('.')[0]

        metadata = DATA.metadata(data_home)
        self.metadata = metadata[track_id]

    def __repr__(self):
        repr_string = "Eurovision Track(track_id={}, audio_path={}, title={})"
        return repr_string.format(self.track_id, self.audio_path, "")  # self.title)

    @property
    def audio(self):
        return load_audio(self.audio_path)


def load_audio(audio_path):
    """Load a Eurovision audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    return librosa.load(audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False):
    """Download the Eurovision Dataset (annotations).
    The audio files are not provided due to the copyright.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool): Whether to overwrite the existing downloaded data

    """

    # use the default location: ~/mir_datasets/Eurovision
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_message = """
        Unfortunately the audio files of the Eurovision dataset are not available
        for download. If you have the Eurovision dataset, place the contents into
        a folder called Eurovision with the following structure:
            > Eurovision/
                > annotations/
                > audio/
        and copy the Eurovision folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        file_downloads=[CONTESTANTS_REMOTE, VOTES_REMOTE],
        info_message=download_message,
        force_overwrite=force_overwrite,
    )


def validate(data_home=None, silence=False):
    """Validate if a local version of this dataset is consistent

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths where the expected file exists locally
            but has a different checksum than the reference

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Get the list of track IDs for this dataset

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load Eurovision dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    eurovision_data = {}
    for key in track_ids():
        eurovision_data[key] = Track(key, data_home=data_home)
    return eurovision_data


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========


========== Bibtex ==========

    """

    print(cite_data)
