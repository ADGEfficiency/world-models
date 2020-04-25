import os

import boto3
import tensorflow as tf

from worldmodels.params import home


def make_directories(*dirs):
    """ make many directories at once """
    [os.makedirs(os.path.join(home, d), exist_ok=True) for d in dirs]


def calc_batch_per_epoch(
    epochs, batch_size, records, samples_per_record=1
):
    """ used in vae & memory training """
    print('training of {} epochs'.format(epochs))
    batch_per_epoch = int(samples_per_record * len(records) / batch_size)
    print('{} batches per epoch'.format(batch_per_epoch))
    return epochs, batch_size, batch_per_epoch


def list_records(
    path, contains, data
):
    """ interface to S3 or local files """
    if str(data).lower() == 's3':
        return list_s3_objects(contains)
    elif data == 'local':
        return list_local_files(path, contains)
    else:
        raise ValueError('data source {} not recognized'.format(data))


def list_s3_objects(contains):
    """ interface to S3 """
    print('S3 objects that include {}'.format(contains))
    s3 = boto3.resource('s3')
    name = 'world-models'
    bucket = s3.Bucket(name)
    objs = bucket.objects.all()
    objs = [o for o in objs if contains in o.key]
    print('found {} objects'.format(objs))
    return sorted(['s3://{}/{}'.format(name, o.key) for o in objs])


def list_local_files(record_dir, incl):
    """ interface to local files """
    print('local files that contain {} in {}'.format(incl, record_dir))
    record_dir = os.path.join(home, record_dir)
    files = os.listdir(record_dir)
    files = sorted([os.path.join(record_dir, f) for f in files if incl in f])
    print('found {} files'.format(len(files)))
    return files


def validate_dataset(filenames, reader_opts=None):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    """
    i = 0
    for fname in filenames:
        print('validating ', fname)

        record_iterator = tf.io.tf_record_iterator(path=fname, options=reader_opts)
        try:
            for _ in record_iterator:
                i += 1
        except Exception as e:
            print('error in {} at record {}'.format(fname, i))
            print(e)


if __name__ == '__main__':
    from worldmodels.data.upload_to_s3 import list_local_records
    from worldmodels.data.tf_records import parse_random_rollouts

    records = list_local_records('random-rollouts', 'episode')

    for record in records:
        print(record)
        for _ in tf.data.TFRecordDataset(record).map(parse_random_rollouts).take(1):
            pass
