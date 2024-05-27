import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split


def get_train_transforms(input_shape, image_size):
    return transforms.Compose([
        transforms.ToTensor(),  # Convert the image back to a tensor
        transforms.Resize((input_shape[0] + 20, input_shape[0] + 20)),  # Resize to INPUT_SHAPE + 20
        transforms.RandomCrop((image_size, image_size)),  # Random crop to IMAGE_SIZE
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])


def get_test_transforms(image_size):
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((image_size, image_size)),  # Resize to IMAGE_SIZE
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def prepare_data_cifar(data_dir=None, input_shape=None, image_size=None, batch_size=None):
    train_transform = get_train_transforms(input_shape, image_size)
    test_transform = get_test_transforms(image_size)

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # only pick 256 samples for local debugging
    # train_set = Subset(train_set, list(range(256)))
    # test_set = Subset(test_set, list(range(256)))

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_set, batch_size, shuffle=False, num_workers=1)

    return train_dataloader, test_dataloader, train_set, test_set
