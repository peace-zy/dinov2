# Dinov2 Pretraining with Skypilot

The original dinov2 repository provides a script to run pretraining on a labeled ImageNet style dataset in a SLURM cluster.

If you, like me, don't have access to a SLURM cluster and/or your image dataset doesn't have labels(isn't that why we're doing SSL?), then this fork is for you!

Note that you'll definitely need to edit some of the config files here to work with your dataset.
- sky/vitl-p14-bs1024-ep10.yaml
- dinov2/configs/train/vitb14_ep10.yaml


## Prerequisites

### Skypilot

In order to run dinov2 pretraining, you'll need [skypilot installed](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html).
I recommend you install from source, but it's not required.

Because the dinov2 hyperparameters are set up for batch size 1024,
you'll need to configure the gpus * batch_size_per_gpu = 1024.

Unless you've run large ML training jobs before, you'll almost certainly need to [request a quota increase](https://skypilot.readthedocs.io/en/latest/cloud-setup/quota.html?highlight=quota)
from your cloud provider. My GCP quota increase was approved basically instantly, but I'm still waiting on AWS...
If your training deployment is hanging or cycling(spinning up and down the head node), you probably don't have enough GPU quota in that region.

### Code

Make sure you're inside this directory when running `sky launch/sky exec` since workdir is relative to the current working directory.

```bash
git clone https://github.com/dinov2.git
cd dinov2/sky
```

### Dataset
Everyone keeps their datasets in different formats so you'll need to modify vitl-p14-bs1024-ep10.yaml appropriately to pull down your dataset.

MAKE SURE THAT ALL OF YOUR IMAGES ARE VALID! I had a few corrupted images in my dataset and training will proceed until it hits a corrupted image and then crash.

```python
from fastai.vision.all import *
images = get_image_files(Path("~/.datasets/").expanduser())

# Unfortunately, there's no progress bar, but this runs in parallel and ran at ~100 images/s/cpu for me.
invalid = verify_images(images)
print("Invalid Images:", invalid)
```

If you naively store your image dataset as millions of little jpg or png files in a GCS bucket, it will take approximately forever to download them with all with `gsutil -m cp/rsync -r` which is what skypilot uses for its
[storage COPY mode](https://skypilot.readthedocs.io/en/latest/reference/storage.html). Instead, you can install
rclone and edit the bucket names inside `vitl-p14-bs1024-ep10.yaml`.

This skypilot team did a great job of throwing up a tiny benchmark for this: https://github.com/skypilot-org/skypilot/issues/2771

If you keep your images in a collection of right-sized tarballs, skypilot's COPY mode will probably work great. Just remember to extract them into dataset_path(~/.datasets/ by default).

You can change the dataset path in `dinov2/configs/train/vitb14_ep10.yaml`.


## Usage

To run dinov2 pretraining with skypilot:

1. Make sure you are in the `dinov2/sky` directory if you aren't already:

    ```bash
    cd sky
    ```

2. Make sure you've properly edited the config files to work with your dataset.

    In order to get this setup initially, you'll probably want to change the hardware to something cheaper and then
    start an unmanaged spot instance or on-demand instance. Managed spot instances are great, but they teardown
    when a job is complete or crashes. This is fine for training, but not so great for debugging.

    ```bash
    sky launch -c dinov2 vitl-p14-bs1024-ep10.yaml # On-demand
    sky launch -c dinov2 vitl-p14-bs1024-ep10.yaml --use-spot # Unmanaged spot
    ```

    Unmanaged spot instances can still be interrupted by the cloud provider, but on GCP their disk will persist so relaunching it is pretty fast.

    You can ssh into your instance to run commands with:

    ```bash
    ssh dinov2
    ```

    Or with `sky exec`. Check out the [skypilot docs](https://skypilot.readthedocs.io/en/latest/reference/cli.html?highlight=exec#sky-exec) for more info.


2. Run the pretraining script with skypilot:

    ```bash
    sky spot launch vitl-p14-bs1024-ep10.yaml
    ```

3. Wait for the pretraining process to complete. This may take a while depending on the hardware you are using.
For me, training took about 7 hours for 10 epochs on 8 A100 GPUs.

4. Once the pretraining is finished, you can find the trained model weights in the specified output directory.

5. If everything went well and you have spare GPU hours, you can try to train for more epochs by modifying the configuration file and running the pretraining script again. The original README suggested training for 500 epochs, but I haven't tried that yet as it's just too expensive.

## Notes and Gotchas

### Epoch Definition

The original dinov2 repository redefined the term "epoch" to mean "a specific number of iterations(1250 by default)" instead of "number of passes over the dataset". This means that (all else being equal) adding more GPUs won't speed up training, it will increase the effective batch size which increases the number images per epoch.

Assuming a batch size of 1024 and OFFICIAL_EPOCH_LENGTH of 1250, the number of images per epoch is 1024 * 1250 = 1,280,000.

See https://github.com/facebookresearch/dinov2/issues/135#issuecomment-1620092479

### Which GPUs to Use

You can just-barely fit bs=128 on an A100-80GB. I've only tried this with FSDP over 8 GPUs, so not sure if it will OOM on 1 GPU.

bs=64 won't fit on an A100-40GB, so your next-best option is 32xA100-40GB w/ bs=32. While this seems like it would be dramatically more expensive than 8xA100-80GB w/ bs=128, the 32xA100-40GB trains ~4x faster.

GCP  8xA100-80GB: ~$14/hr * ~7hrs = ~$98
GCP 32xA100-40GB: ~$47/hr * ~1.75hrs = ~$82

I didn't actually try out the 32xA100-40GB configuration, but it might actually turn out to be cheaper, even on GCP.
