from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms


################################################################################
# Transforms to be used on images
################################################################################
TRAINING_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor()
    ]
)


################################################################################
# Dataset Objects
################################################################################
class SignDataset(Dataset):
    """
    SignDataset is an object that holds a dataset of images representing
    Signed Language gestures.
    """

    def __init__(self, data_dir: str, transform):
        """
        Args:
            data_dir: directory containing data files/images.
            transform: transform functions applied to images.
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, file) for file in self.filenames]
        # First char in each image file name is the label
        self.labels = [int(filename.split('/')[-1][0]) for filename in self.filenames]
        # Transform used on image on get
        self.transform = transform

    def __getitem__(self, index: int):
        image = self.transform(Image.open(self.filenames[index]))
        return image, self.labels[index]

    def __len__(self):
        return len(self.filenames)


def get_data_loader(data_dir: str, params, training=False, testing=False, eval=False):
    """
    Fetches the data loader for a particular Dataset.

    Args:
        data_dir: directory containing data.
        params: hyperparameters for the dataset.
        training: True if this is a training set.
        testing: True if this is a testing set.
        eval: True if this is an evaluation set.

    Returns:
        DataLoader: Data loader for a dataset, else None type object.
    """
    data_loader = None
    if training:
        path = os.path.join(data_dir, "train_signs")
        data_loader = DataLoader(
            SignDataset(path, TRAINING_TRANSFORM),
            batch_size=params.batch_size, shuffle=True,
            num_workers=params.num_workers, pin_memory=params.cuda
        )
    elif testing or eval:
        if testing:
            path = os.path.join(data_dir, "test_signs")
        else:
            path = os.path.join(data_dir, "eval_signs")
        data_loader = DataLoader(
            SignDataset(path, EVAL_TRANSFORM),
            batch_size=params.batch_size, shuffle=True,
            num_workers=params.num_workers, pin_memory=params.cuda
        )
    return data_loader

