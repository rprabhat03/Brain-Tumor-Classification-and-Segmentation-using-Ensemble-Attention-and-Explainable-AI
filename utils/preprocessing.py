"""
Preprocessing utilities for Brain Tumor Classification & Segmentation.
Handles image loading, resizing, augmentation, dataset merging, and deduplication.
"""

import os
import shutil
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
except ImportError:
    print("Warning: imagehash not installed. Run: pip install imagehash")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = 224
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_ALIASES = {
    'glioma': 'glioma',
    'glioma_tumor': 'glioma',
    'meningioma': 'meningioma',
    'meningioma_tumor': 'meningioma',
    'no_tumor': 'notumor',
    'notumor': 'notumor',
    'no tumor': 'notumor',
    'normal': 'notumor',
    'healthy': 'notumor',
    'pituitary': 'pituitary',
    'pituitary_tumor': 'pituitary',
}


def normalize_class_name(name):
    """Normalize class folder name to one of the 4 standard classes."""
    normalized = name.lower().strip().replace(' ', '_')
    if normalized in CLASS_ALIASES:
        return CLASS_ALIASES[normalized]
    # Fuzzy matching
    for key, val in CLASS_ALIASES.items():
        if key in normalized or normalized in key:
            return val
    return None


def load_and_resize_image(path, size=IMG_SIZE):
    """Load an image and resize to (size, size). Returns PIL Image or None."""
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def compute_perceptual_hash(img, hash_size=16):
    """Compute perceptual hash for near-duplicate detection."""
    return str(imagehash.phash(img, hash_size=hash_size))


def compute_file_hash(path):
    """Compute MD5 hash of file contents for exact-duplicate detection."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_dataset_folder(root_dir):
    """
    Scan a dataset folder for images organized by class subfolders.
    Returns dict: {class_name: [list of absolute image paths]}
    """
    result = {}
    if not os.path.isdir(root_dir):
        print(f"Warning: {root_dir} does not exist")
        return result

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        class_name = normalize_class_name(folder_name)
        if class_name is None:
            # Check if this is a split folder (Training/Testing)
            for sub_folder in os.listdir(folder_path):
                sub_path = os.path.join(folder_path, sub_folder)
                if os.path.isdir(sub_path):
                    sub_class = normalize_class_name(sub_folder)
                    if sub_class:
                        if sub_class not in result:
                            result[sub_class] = []
                        for img_file in os.listdir(sub_path):
                            img_path = os.path.join(sub_path, img_file)
                            if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                                result[sub_class].append(img_path)
        else:
            if class_name not in result:
                result[class_name] = []
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    result[class_name].append(img_path)

    return result


def merge_datasets(dataset_dirs, output_dir, size=IMG_SIZE, use_phash=True, hash_size=16):
    """
    Merge multiple dataset directories into a single deduplicated dataset.

    Args:
        dataset_dirs: list of paths to raw dataset directories
        output_dir: path to output merged directory (will have train/test subfolders)
        size: target image size
        use_phash: use perceptual hashing for dedup (else MD5)
        hash_size: perceptual hash size (higher = more strict)

    Returns:
        dict with merge statistics
    """
    from sklearn.model_selection import train_test_split

    all_images = {cls: [] for cls in CLASSES}
    seen_hashes = set()
    stats = {
        'total_scanned': 0,
        'duplicates_removed': 0,
        'per_class': {},
        'per_dataset': {}
    }

    # Step 1: Scan all datasets
    for i, ds_dir in enumerate(dataset_dirs):
        ds_name = os.path.basename(ds_dir)
        ds_images = scan_dataset_folder(ds_dir)
        ds_count = 0

        for cls, paths in ds_images.items():
            for path in tqdm(paths, desc=f"Processing {ds_name}/{cls}"):
                stats['total_scanned'] += 1
                img = load_and_resize_image(path, size)
                if img is None:
                    continue

                # Compute hash for dedup
                if use_phash:
                    h = compute_perceptual_hash(img, hash_size)
                else:
                    h = compute_file_hash(path)

                if h in seen_hashes:
                    stats['duplicates_removed'] += 1
                    continue

                seen_hashes.add(h)
                all_images[cls].append((path, img))
                ds_count += 1

        stats['per_dataset'][ds_name] = ds_count

    # Step 2: Split and save
    for cls in CLASSES:
        images = all_images[cls]
        if len(images) == 0:
            print(f"Warning: No images found for class '{cls}'")
            continue

        # 80/20 train/test split
        train_imgs, test_imgs = train_test_split(
            images, test_size=0.2, random_state=42
        )

        train_dir = os.path.join(output_dir, 'train', cls)
        test_dir = os.path.join(output_dir, 'test', cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for idx, (orig_path, img) in enumerate(train_imgs):
            img.save(os.path.join(train_dir, f"{cls}_train_{idx:05d}.jpg"), quality=95)

        for idx, (orig_path, img) in enumerate(test_imgs):
            img.save(os.path.join(test_dir, f"{cls}_test_{idx:05d}.jpg"), quality=95)

        stats['per_class'][cls] = {
            'total': len(images),
            'train': len(train_imgs),
            'test': len(test_imgs)
        }

    return stats


def get_data_generators(data_dir, img_size=IMG_SIZE, batch_size=32, validation_split=0.2):
    """
    Create train, validation, and test data generators using ImageDataGenerator.

    Args:
        data_dir: path to classification directory with train/ and test/ subfolders
        img_size: target image size
        batch_size: batch size
        validation_split: fraction of training data for validation

    Returns:
        (train_gen, val_gen, test_gen)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def get_segmentation_data(images_dir, masks_dir, img_size=256, test_split=0.2):
    """
    Load segmentation images and masks, split into train/test.

    Returns:
        (X_train, X_test, y_train, y_test) as numpy arrays
    """
    from sklearn.model_selection import train_test_split

    images = []
    masks = []

    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    for img_f, mask_f in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Loading segmentation data"):
        img = Image.open(os.path.join(images_dir, img_f)).convert('RGB')
        img = img.resize((img_size, img_size), Image.LANCZOS)
        images.append(np.array(img) / 255.0)

        mask = Image.open(os.path.join(masks_dir, mask_f)).convert('L')
        mask = mask.resize((img_size, img_size), Image.NEAREST)
        masks.append(np.array(mask) / 255.0)

    X = np.array(images, dtype=np.float32)
    y = np.array(masks, dtype=np.float32)
    y = np.expand_dims(y, axis=-1)  # (N, H, W, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )

    return X_train, X_test, y_train, y_test
