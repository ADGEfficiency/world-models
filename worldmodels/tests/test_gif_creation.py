from worldmodels.vision.images import sort_image_files


def test_get_image_files():
    """ check we sort a list of image files """
    images = [
        'epoch_1_batch_2.png',
        'epoch_1_batch_1.png',
        'epoch_1_batch_20.png',
        'epoch_1_batch_100.png',
        'epoch_2_batch_2.png',
        'epoch_2_batch_1.png'
    ]

    expected = [
        'epoch_1_batch_1.png',
        'epoch_1_batch_2.png',
        'epoch_1_batch_20.png',
        'epoch_1_batch_100.png',
        'epoch_2_batch_1.png',
        'epoch_2_batch_2.png',
    ]

    new_images = sort_image_files(images)
    assert new_images == expected
