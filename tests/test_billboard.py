# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os
import pytest
from mirdata import billboard, utils
from tests.test_utils import mock_validator, DEFAULT_DATA_HOME
from tests.test_download_utils import mock_downloader


def test_track():
    # test data home None
    track_default = billboard.Track('3')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'McGill-Billboard')

    # test specific data home
    data_home = 'tests/resources/mir_datasets/McGill-Billboard'

    with pytest.raises(ValueError):
        billboard.Track('asdfasdf', data_home=data_home)

    track = billboard.Track('3', data_home=data_home)

    # test attributes are loaded as expected
    assert track.track_id == '3'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': ['audio/1960s/James Brown/I Don\'t Mind/audio.flac', 'bb9f022b25c43983cf19aef562b00eac'],
        'salami': [
            'annotations/0003/salami_chords.txt',
            '8deb413e4cecadcffa5a7180a5f4c597',
        ]
    }

    # assert track.audio_path == 'tests/resources/mir_datasets/McGill-Billboard/' + 'audio/1960s/James Brown/I Don\'t Mind/audio.flac'
    assert track.title == 'I Don\'t Mind'
    assert track.artist == 'James Brown'

    # # test that cached properties don't fail and have the expected type
    assert type(track.sections) is utils.SectionData

    # # test audio loading functions
    # y, sr = track.audio
    # assert sr == 44100
    # assert y.shape == (89856,)


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/McGill-Billboard'
    track = billboard.Track('3', data_home=data_home)
    jam = track.to_jams()

    segments = jam.search(namespace='segment')[0]['data']
    print([segment.time for segment in segments])
    assert [segment.time for segment in segments] == [0.0,
        0.073469387,
        22.346394557,
        49.23802721,
        69.942857142,
        76.123990929,
        96.418956916,
        102.924353741,
        123.887596371,
        130.206598639,
        148.723809523,
        150.909387755
    ]
    
    assert [segment.duration for segment in segments] == [
        0.073469387, 
        8.640544218,
        6.873038549,
        6.9470521550000015,
        6.181133787000007,
        6.687210884999999,
        6.505396825000005,
        6.962630384999997,
        6.319002267999991,
        6.83292517000001,
        2.1855782320000117,
        0.0
    ]
    
    assert [segment.value for segment in segments] == [
        None,
        ('A', 'intro'),
        ('B', 'verse'),
        ('B', 'verse'),
        None,
        ('A', 'interlude'),
        None,
        ('B', 'verse'),
        None,
        ('A', 'interlude'),
        None,
        None
    ]
    
    assert [segment.confidence for segment in segments] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    ]

    assert jam['file_metadata']['title'] == 'I Don\'t Mind'
    assert jam['file_metadata']['artist'] == 'James Brown'


def test_track_ids():
    track_ids = billboard.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 890


def test_load():
    data_home = 'tests/resources/mir_datasets/McGill-Billboard'
    data = billboard.load(data_home=data_home)
    assert type(data) is dict
    assert len(data.keys()) == 890

    # data home default
    data_default = billboard.load()
    assert type(data_default) is dict
    assert len(data_default.keys()) == 890


def test_load_sections():
    # load a file which exists
    sections_path = (
        'tests/resources/mir_datasets/McGill-Billboard/'
        + 'annotations/0035/salami_chords.txt'
    )
    section_data = billboard._load_sections(sections_path)

    # check types
    assert type(section_data) == utils.SectionData
    assert type(section_data.intervals) is np.ndarray
    assert type(section_data.labels) is list

    # check valuess

    assert np.array_equal(
        section_data.labels, np.array([None, None, ("A'", 'intro'), None, ('A', 'verse'), ('B', 'chorus'), None, ('C', 'solo'), None, ('A', 'verse'), ('B', 'chorus'), None, ('D', 'trans'), ('E', 'bridge'), None, ('F', 'solo'), ("A'", 'verse'), None, ('B', 'chorus'), None, ('G', 'outro'), None, ('Z', 'fadeout'), None, None])
    )

    # load a file which doesn't exist
    section_data_none = billboard._load_sections('fake/file/path')
    assert section_data_none is None

    # load none
    section_data_none2 = billboard._load_sections('asdf/asdf')
    assert section_data_none2 is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/McGill-Billboard'
    metadata = billboard._load_metadata(data_home)
    assert metadata['3'] == {
        'title': 'I Don\'t Mind',
        'artist': 'James Brown',
        'actual_rank': 57,
        'peak_rank': 47,
        'target_rank': 56,
        'weeks_on_chart': 8,
        'chart_date': '1961-07-03'
    }

    none_metadata = billboard._load_metadata('asdf/asdf')
    assert none_metadata is None


def test_download(mock_downloader):
    billboard.download()
    mock_downloader.assert_called()


def test_validate():
    billboard.validate()
    billboard.validate(silence=True)


def test_cite():
    billboard.cite()