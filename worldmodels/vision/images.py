from collections import defaultdict
import os
import re

import imageio
import matplotlib.pyplot as plt


def compare_images(model, sample_observations, image_dir):
    """ side by side comparison of image and reconstruction """
    reconstructed = model.forward(sample_observations)

    fig, axes = plt.subplots(
        nrows=sample_observations.shape[0],
        ncols=2,
        figsize=(5, 8)
    )

    for idx in range(sample_observations.shape[0]):
        actual_ax = axes[idx, 0]
        reconstructed_ax = axes[idx, 1]

        actual_ax.imshow(sample_observations[idx, :, :, :])
        reconstructed_ax.imshow(reconstructed[idx, :, :, :])
        actual_ax.set_axis_off()
        reconstructed_ax.set_axis_off()

        actual_ax.set_aspect('equal')
        reconstructed_ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(image_dir, 'compare.png'))


def generate_images(model, epoch, batch, sample_latent, image_dir):
    """ latent to reconstructed images """
    assert sample_latent.shape[0] == 4
    predictions = model.decode(sample_latent)
    fig, axes = plt.subplots(figsize=(4, 4), nrows=2, ncols=2)

    axes = axes.reshape(-1)
    for idx, ax in enumerate(axes):
        ax.imshow(predictions[idx])
        ax.axis('off')

    plt.savefig('{}/epoch_{}_batch_{}.png'.format(image_dir, epoch, batch))


def sort_image_files(image_list):
    """ orders the images generated during training """
    epochs = defaultdict(list)

    #  group into epochs
    max_epoch = 0
    for image in image_list:
        epoch = re.search(r'epoch_([0-9]*)_(.*)', image).groups()[0]
        epochs[epoch].append(image)
        max_epoch = max(int(epoch), max_epoch)

    #  sort each of the lists
    sorted_batches = []
    for epoch in range(1, max_epoch+1):

        batch = epochs[str(epoch)]

        sort_array = []
        for image in batch:
            sort_array.append(int(re.search(r'batch_([0-9]+)', image).groups()[0]))

        sorted_batch = [image for idx, image in sorted(zip(sort_array, batch), reverse=False)]

        for batch in sorted_batch:
            sorted_batches.append(batch)

    return sorted_batches


def generate_gif(image_dir, output_dir):
    print('generating gif from images in {}'.format(image_dir))

    image_list = [x for x in os.listdir(image_dir) if '.png' in x]
    image_files = sort_image_files(image_list)
    image_files = [os.path.join(image_dir, x) for x in image_list]
    image_files = [imageio.imread(f) for f in image_files]

    anim_file = os.path.join(output_dir, 'training.gif')
    imageio.mimsave(anim_file, image_files)
