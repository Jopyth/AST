from typing import List
import albumentations as A


from enum import Enum
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
# from anomalib.data.utils.image import get_image_height_and_width


# data augmentation code from https://github.com/scortexio/patchcore-few-shot
class Transformer:
    def __init__(self, dataset_name):
        affine = A.Affine(
            translate_px=(-16, 16),
            rotate=(-5, 5),
            scale=(0.95, 1.05),
            p=0.25,
        )

        random_brightness_contrast = A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.25,
        )

        blur = A.Blur(
            blur_limit=3,
            p=0.25,
        )

        sharpen = A.Sharpen(
            alpha=(0.1, 0.3),
            lightness=(0.5, 1.0),
            p=0.25,
        )

        flip = A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.25,
        )

        # if dataset_name == "visa":
        #     self.transforms = [affine, random_brightness_contrast, blur, sharpen, flip]  # change for MVTedc
        
        if dataset_name == "mvtec":
            self.transforms = [affine, random_brightness_contrast, blur]

        # else:
        #     raise ValueError(f"Dataset {dataset_name} is not supported")

    def get_transforms_with_index(self, list_index: List[int]) -> A:
        return A.Compose(
            A.OneOf(
                [self.transforms[index] for index in list_index],
                p=1.0,
            )
        )


def get_augmentation_combinations_from_transformer(transformer: Transformer) -> List[List[int]]:
    """Given a transformer which contains a list of augmentations. Return the indexes of different
    augmentation combination, concretely:
    + Indexes of all augmentations
    + Indexes of all augmentations after removed 1 augmentation.

    E.g Given augmentations = [A, B, C, D]
    => augmentation_combination = [
        [0, 1, 2, 3],
        [   1, 2, 3],
        [0,    2, 3],
        [0, 1,  , 3]
        [0, 1, 2,  ],

    ]

    Args:
        transformer (Transformer): the transformer which contains list of augmentations
    """
    all_indexes = [idx for idx in range(len(transformer.transforms))]
    print(f"all indexes: {all_indexes}")
    combinations = [all_indexes]
    for idx in all_indexes:
        all_indexes_copy = all_indexes.copy()
        all_indexes_copy.remove(idx)
        combinations.append(all_indexes_copy)
    print(f"All combinations: {combinations}")

    return combinations


# if transform methods for data augmentation are not defined, we will provide some dafault transforms.

class InputNormalizationMethod(str, Enum):
    """Normalization method for the input images."""

    NONE = "none"  # no normalization applied
    IMAGENET = "imagenet"  # normalization to ImageNet statistics


def get_image_height_and_width(image_size):
    """Get image height and width from ``image_size`` variable.

    Args:
        image_size (int | tuple[int, int] | None, optional): Input image size.

    Raises:
        ValueError: Image size not None, int or tuple.

    Examples:
        >>> get_image_height_and_width(image_size=256)
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256))
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256, 3))
        (256, 256)

        >>> get_image_height_and_width(image_size=256.)
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_height_and_width
        ValueError: ``image_size`` could be either int or tuple[int, int]

    Returns:
        tuple[int | None, int | None]: A tuple containing image height and width values.
    """
    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        height_and_width = int(image_size[0]), int(image_size[1])
    else:
        raise ValueError("``image_size`` could be either int or tuple[int, int]")

    return height_and_width


def get_transforms(
    config = None,
    image_size = None,
    center_crop = None,
    normalization = InputNormalizationMethod.IMAGENET,
    to_tensor = True,
) -> A.Compose:
    """Get transforms from config or image size.

    Args:
        config (str | A.Compose | None, optional): Albumentations transforms.
            Either config or albumentations ``Compose`` object. Defaults to None.
        image_size (int | tuple | None, optional): Image size to transform. Defaults to None.
        to_tensor (bool, optional): Boolean to convert the final transforms into Torch tensor. Defaults to True.

    Raises:
        ValueError: When both ``config`` and ``image_size`` is ``None``.
        ValueError: When ``config`` is not a ``str`` or `A.Compose`` object.

    Returns:
        A.Compose: Albumentation ``Compose`` object containing the image transforms.

    Examples:
        >>> import skimage
        >>> image = skimage.data.astronaut()

        >>> transforms = get_transforms(image_size=256, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (256, 256, 3)

        >>> transforms = get_transforms(image_size=256, to_tensor=True)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 256, 256])


        Transforms could be read from albumentations Compose object.
        >>> import albumentations as A
        >>> from albumentations.pytorch import ToTensorV2
        >>> config = A.Compose([A.Resize(512, 512), ToTensorV2()])
        >>> transforms = get_transforms(config=config, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (512, 512, 3)
        >>> type(output["image"])
        numpy.ndarray

        Transforms could be deserialized from a yaml file.
        >>> transforms = A.Compose([A.Resize(1024, 1024), ToTensorV2()])
        >>> A.save(transforms, "/tmp/transforms.yaml", data_format="yaml")
        >>> transforms = get_transforms(config="/tmp/transforms.yaml")
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 1024, 1024])
    """
    transforms: A.Compose

    if config is not None:
        if isinstance(config, DictConfig):
            logger.info("Loading transforms from config File")
            transforms_list = []
            for key, value in config.items():
                if hasattr(A, key):
                    transform = getattr(A, key)(**value)
                    logger.info(f"Transform {transform} added!")
                    transforms_list.append(transform)
                else:
                    raise ValueError(f"Transformation {key} is not part of albumentations")

            transforms_list.append(ToTensorV2())
            transforms = A.Compose(transforms_list, additional_targets={"image": "image", "depth_image": "image"})

        # load transforms from config file
        elif isinstance(config, str):
            logger.info("Reading transforms from Albumentations config file: %s.", config)
            transforms = A.load(filepath=config, data_format="yaml")
        elif isinstance(config, A.Compose):
            logger.info("Transforms loaded from Albumentations Compose object")
            transforms = config
        else:
            raise ValueError("config could be either ``str`` or ``A.Compose``")
    else:
        logger.info("No config file has been provided. Using default transforms.")
        transforms_list = []

        # add resize transform
        if image_size is None:
            raise ValueError(
                "Both config and image_size cannot be `None`. "
                "Provide either config file to de-serialize transforms "
                "or image_size to get the default transformations"
            )
        resize_height, resize_width = get_image_height_and_width(image_size)
        transforms_list.append(A.Resize(height=resize_height, width=resize_width, always_apply=True))

        # add center crop transform
        if center_crop is not None:
            crop_height, crop_width = get_image_height_and_width(center_crop)
            if crop_height > resize_height or crop_width > resize_width:
                raise ValueError(f"Crop size may not be larger than image size. Found {image_size} and {center_crop}")
            transforms_list.append(A.CenterCrop(height=crop_height, width=crop_width, always_apply=True))

        # add normalize transform
        if normalization == InputNormalizationMethod.IMAGENET:
            transforms_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        elif normalization == InputNormalizationMethod.NONE:
            transforms_list.append(A.ToFloat(max_value=255))
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

        # add tensor conversion
        if to_tensor:
            transforms_list.append(ToTensorV2())

        transforms = A.Compose(transforms_list, additional_targets={"image": "image", "depth_image": "image"})

    return transforms

