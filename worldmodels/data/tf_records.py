import os

import tensorflow as tf


def encode_float(value):
    """ single array """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def encode_floats(features):
    """ multiple arrays """
    package = {}
    for key, value in features.items():
        package[key] = encode_float(value.flatten().tolist())

    example_proto = tf.train.Example(features=tf.train.Features(feature=package))
    return example_proto.SerializeToString()


def save_episode_tf_record(results_dir, results, process_id, episode):
    """ results dictionary to .tfrecord """

    path = os.path.join(
        results_dir,
        'process{}-episode{}.tfrecord'.format(process_id, episode)
    )

    print('saving to {}'.format(path))
    with tf.io.TFRecordWriter(path) as writer:
        for obs, act in zip(results['observation'], results['action']):
            encoded = encode_floats({'observation': obs, 'action': act})
            writer.write(encoded)


def parse_episode(example_proto):
    """ used in training VAE """
    features = {
        'observation': tf.io.FixedLenFeature((64, 64, 3), tf.float32),
        'action': tf.io.FixedLenFeature((3,), tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features['observation'], parsed_features['action']


def parse_latent_stats(example_proto):
    """ used in training memory """
    features = {
        'action': tf.io.FixedLenFeature((1000, 3,), tf.float32),
        'mu': tf.io.FixedLenFeature((1000, 32,), tf.float32),
        'logvar': tf.io.FixedLenFeature((1000, 32,), tf.float32)
    }
    return tf.io.parse_single_example(example_proto, features)


def shuffle_samples(
        parse_func,
        records_list,
        batch_size,
        repeat=None,
        shuffle_buffer=5000,
        num_cpu=8,
):
    """ used in vae training """
    files = tf.data.Dataset.from_tensor_slices(records_list)

    #  get samples from different files
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=num_cpu,
        cycle_length=num_cpu
    )
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(parse_func, num_parallel_calls=num_cpu)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat).prefetch(1)
    return iter(dataset)


def batch_episodes(parse_func, records, episode_length, num_cpu=4):
    """ used in sampling latent stats """
    files = tf.data.Dataset.from_tensor_slices(records)

    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=num_cpu,
        cycle_length=num_cpu,
        block_length=episode_length
    )
    dataset = dataset.map(parse_func, num_parallel_calls=num_cpu)
    dataset = dataset.batch(episode_length)
    dataset = dataset.repeat(None)
    return iter(dataset)
