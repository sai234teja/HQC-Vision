import os
import cv2
import numpy as np
import pandas as pd
from glob import glob


def _series_uid_from_csv_path(csv_path: str) -> str:
    """
    Extracts the Series UID (= JPEG filename stem) from a CBIS-DDSM CSV path.

    CSV paths look like:
      Calc-Test_P_00038_LEFT_CC/<StudyUID>/<SeriesUID>/000000.dcm
    The SeriesUID at index [2] matches the JPEG filenames in images/ and masks/.
    """
    parts = csv_path.strip().split('/')
    if len(parts) >= 3:
        return parts[2]
    return None


class DataLoader:
    """
    CBIS-DDSM aware loader.

    Builds paired (image, mask) entries by joining the calc CSV manifests
    on the Series UID embedded in each file path column.
    Only calcification cases are loaded (matching this project's focus).
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir  = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir  = os.path.join(data_dir, 'masks')
        self.csv_dir   = os.path.join(data_dir, 'csv')

        # Build lookup: SeriesUID -> full file path on disk
        self._img_index  = self._index_folder(self.image_dir)
        self._mask_index = self._index_folder(self.mask_dir)

        # Load the correct CSV split
        split_tag = 'train' if split == 'train' else 'test'
        csv_path  = os.path.join(
            self.csv_dir, f'calc_case_description_{split_tag}_set.csv'
        )
        df = pd.read_csv(csv_path)

        self.pairs   = []
        self.skipped = 0

        for _, row in df.iterrows():
            img_uid  = _series_uid_from_csv_path(str(row['image file path']))
            mask_uid = _series_uid_from_csv_path(str(row['ROI mask file path']))

            img_path  = self._img_index.get(img_uid)
            mask_path = self._mask_index.get(mask_uid)

            if img_path is None or mask_path is None:
                self.skipped += 1
                continue

            self.pairs.append({
                'image_path': img_path,
                'mask_path':  mask_path,
                'patient_id': row['patient_id'],
                'pathology':  row['pathology'],
                'assessment': row['assessment'],
                'calc_type':  row.get('calc type', ''),
                'subtlety':   row['subtlety'],
            })

        print(f"[DataLoader] split={split_tag} | "
              f"matched={len(self.pairs)} pairs | "
              f"skipped={self.skipped} (file not on disk)")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def load_pair(self, index: int):
        """Returns (image, mask, metadata_dict) for one case."""
        entry = self.pairs[index]

        image = cv2.imread(entry['image_path'], cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(entry['mask_path'],  cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {entry['image_path']}")
        if mask is None:
            mask = np.zeros_like(image)
        else:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return image, mask, entry

    def load_batch(self):
        """Loads all matched pairs. Returns (images, masks) lists."""
        images, masks = [], []
        for i in range(len(self)):
            img, mask, _ = self.load_pair(i)
            images.append(img)
            masks.append(mask)
        return images, masks

    def __len__(self):
        return len(self.pairs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_folder(self, folder: str) -> dict:
        """Maps SeriesUID stem -> full file path for every jpg/png in folder."""
        index = {}
        for ext in ('*.jpg', '*.png'):
            for fpath in glob(os.path.join(folder, ext)):
                stem = os.path.splitext(os.path.basename(fpath))[0]
                index[stem] = fpath
        return index