import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
import pylidc as pl
import pylidc.utils
from medpy.io import save
from tqdm import tqdm


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path to the folder where the cropped nodules will be stored",
        required=True,
    )
    args = parser.parse_args()
    return args


def has_large_mask(nod):
    """
    Checks if the consensus mask is larger than 64 voxels in any dimension. If this is the case, this nodule is
    filtered out
    """
    consensus_mask, _, _ = pylidc.utils.consensus(nod, clevel=0.1)
    max_size_mask = max(consensus_mask.shape)
    if max_size_mask > 64:
        return True


def append_metadata(metadata_nod, nod, first=False):
    features = [
        "subtlety",
        "internal Structure",
        "calcification",
        "sphericity",
        "margin",
        "lobulation",
        "spiculation",
        "texture",
        "malignancy",
    ]
    if first:
        for feature in features:
            metadata_nod[feature] = []
    if nod is not None:
        for feature in features:
            metadata_nod[feature].append(getattr(nod, feature.replace(" ", "")))
    else:
        for feature in features:
            metadata_nod[feature].append(None)


def save_nodules(args: Namespace):
    # Set up the paths to store the data
    save_path = Path(args.save_path)
    images_save_dir = save_path / "images"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_save_dir, exist_ok=True)

    # Get all the Scans
    scans = pl.query(pl.Scan)
    all_metadata = []
    for scan in tqdm(scans):
        nods = scan.cluster_annotations()
        for nod_idx, nod in enumerate(nods):
            # filter nodules that are larger than 64 voxels in any dimension
            if has_large_mask(nod):
                continue
            metadata_nod = {}
            for ann_idx in range(4):
                if ann_idx == 0:
                    # Scan is only saved for first annotation
                    image_size = 64
                    # resample volume and masks to uniform spacing. Returns the interpolation points to resample the
                    # other annotations the same way.
                    vol, mask, irp_pts = nod[ann_idx].uniform_cubic_resample(
                        image_size - 1, return_irp_pts=True
                    )
                    image_name = f"{str(nod[0].scan.id).zfill(4)}_{str(nod_idx).zfill(2)}.nii.gz"
                    image_save_path = (
                        images_save_dir
                        / image_name
                    )
                    vol = vol[32, :, :]
                    assert vol.shape == (64, 64)
                    save(vol, str(image_save_path))
                    metadata_nod["Study Instance UID"] = str(nod[0].scan.study_instance_uid)
                    metadata_nod["Series Instance UID"] = str(nod[0].scan.series_instance_uid)
                    metadata_nod["Patient ID"] = str(nod[0].scan.patient_id)
                    metadata_nod["Scan ID"] = str(nod[0].scan.id).zfill(4)
                    metadata_nod["Nodule Index"] = str(nod_idx).zfill(2)
                    metadata_nod["Image File Name"] = image_name.split(".")[0]

                if ann_idx < len(nod):
                    annotation = nod[ann_idx]
                else:
                    annotation = None
                if ann_idx == 0:
                    append_metadata(metadata_nod, annotation, first=True)
                else:
                    append_metadata(metadata_nod, annotation)

            metadata_series = pd.Series(metadata_nod)
            all_metadata.append(metadata_series)
    metadata = pd.DataFrame(all_metadata)
    metadata.to_csv(save_path / "annotation_data.csv", index=False)


if __name__ == "__main__":
    cli_args = main_cli()
    save_nodules(cli_args)
