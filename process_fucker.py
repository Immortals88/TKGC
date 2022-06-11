import pkg_resources
import os
import errno
from pathlib import Path
import pickle

import numpy as np

from collections import defaultdict

DATA_PATH = 'data/'


def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(timestamp)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """

    test_ratio = 0.1
    valid_ratio = 0.1


    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id (skip)
    # map train/test/valid with the ids format
    files = ['triples_1', 'triples_2']
    examples = []
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r', encoding='utf-8')
        for line in to_read.readlines():
            lhs, rel, rhs, ts = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(ts)
            try:
                examples.append([lhs, rel, rhs, ts])
                # examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[ts]])
            except ValueError:
                continue
    num_triples = len(examples)
    np.random.shuffle(examples)
    test_part = int(test_ratio * num_triples)
    valid_part = int(valid_ratio * num_triples)
    out = open(Path(DATA_PATH) / name / ('train' + '.pickle'), 'wb')
    pickle.dump(np.array(examples).astype('uint64'), out)
    out.close()

    out2 = open(Path(DATA_PATH) / name / ('test' + '.pickle'), 'wb')
    pickle.dump(np.array(examples[:test_part]).astype('uint64'), out2)
    out2.close()

    out3 = open(Path(DATA_PATH) / name / ('valid' + '.pickle'), 'wb')
    pickle.dump(np.array(examples[test_part: test_part + valid_part]).astype('uint64'), out3)
    out3.close()


    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)
    print("creating filtering lists")

    # create filtering files
    # to_ship.pickle
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    files = ['train', 'valid', 'test']
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    try:
        prepare_dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fuck'), 'fuck')
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(e)
            print("File exists. skipping...")
        else:
            raise

