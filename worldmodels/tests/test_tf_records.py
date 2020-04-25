import os
import shutil

import numpy as np
import tensorflow as tf
import pytest

from worldmodels.data.tf_records import encode_floats
from worldmodels.data.tf_records import batch_episodes, shuffle_samples


home = os.environ['HOME']
results_dir = os.path.join(home, 'world-models-experiments', 'tf-record-testing')
os.makedirs(results_dir, exist_ok=True)


def parse_func(example_proto):
    features = {
        'sample': tf.io.FixedLenFeature((32,), tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features['sample']


def write_test_data(name, episode_len):
    episode = np.random.rand(episode_len, 32).astype(np.float32)

    with tf.io.TFRecordWriter(name) as writer:
        for sample in episode:
            encoded = encode_floats({'sample': sample})
            writer.write(encoded)

    return episode


@pytest.fixture(scope='session')
def make_records():
    records = ['ep0', 'ep1', 'ep2', 'ep3']
    records = [
        os.path.join(results_dir, '{}.tfrecord'.format(name))
        for name in records
    ]

    episode_len = 100
    episodes = [write_test_data(rec, episode_len) for rec in records]

    yield records, episode_len, episodes

    shutil.rmtree(results_dir)


def test_shuffle_samples(make_records):
    records, episode_len, episodes = make_records

    batch_size = 5
    dataset = shuffle_samples(parse_func, records, batch_size=batch_size)

    batches = 100

    history = []
    for num in range(batches):
        batch = next(dataset).numpy()
        assert batch.shape[0] == batch_size
        history.append(np.mean(batch))

    assert np.mean(history) != history[0]
    assert np.mean(history) != history[-1]


def test_batch_tf_records(make_records):
    """ run through single episodes in order """
    records, episode_len, episodes = make_records
    dataset = batch_episodes(parse_func, records, episode_len)

    for ep in episodes:
        batch = next(dataset).numpy()
        np.testing.assert_array_equal(batch[0], ep[0])

    for ep in episodes:
        batch = next(dataset).numpy()
        np.testing.assert_array_equal(batch[0], ep[0])
