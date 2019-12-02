import argparse
import hashlib
import json
import os
import csv

INDEX_PATH = '../mirdata/indexes/billboard_index.json'

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
    with open(file_path, 'rb') as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_index(data_path):
    _index = {}
    index_file = csv.reader(open(os.path.join(data_path, 'billboard-2.0-index.csv')))
    for row in index_file:
        k = row[0]
        _index[k] = row[1:]

    annotations_dir = os.path.join(data_path, 'annotations')
    audio_dir = os.path.join(data_path, 'audio')
    anns = os.listdir(annotations_dir)

    track_ids = []
    index = {}
    txtfiles = []
    for a in anns:
        for t in sorted(os.listdir(os.path.join(annotations_dir, a))):
            if 'txt' in t:
                fp = os.path.join(annotations_dir, a, t)
                track_id = '{}'.format(os.path.basename(a.lstrip('0')))

                if track_id in _index.keys():
                    txtfiles.append(t)
                    track_ids.append(track_id)

                    release_date = _index[track_id][0]
                    track_name = _index[track_id][3]
                    artist = _index[track_id][4]

                    _release_date = '{}s'.format(round(int(release_date.split('-')[0]), -1))
                    audio_path = os.path.join(audio_dir, _release_date, artist, track_name, 'audio.flac')
                    audio_checksum = ''

                    if os.path.exists(audio_path):
                        audio_checksum = md5(audio_path)
                        # print(track_id, artist, track_name, release_date, audio_checksum)
                    else:
                        audio_path = ''

                    annot_checksum = md5(fp)
                    annot_rel = os.path.join('annotations', a, t)
                    audio_rel = os.path.join('audio', _release_date, artist, track_name, 'audio.flac')
                    
                    index[track_id] = {
                        'audio': (audio_rel, audio_checksum),
                        'salami': (annot_rel, annot_checksum)
                    }
              
    with open(INDEX_PATH, 'w') as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make index file.')
    PARSER.add_argument(
        'data_path', type=str, help='Path to data folder.'
    )

    main(PARSER.parse_args())
