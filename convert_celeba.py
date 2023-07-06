import argparse
import os
import multiprocessing
import shutil

import tqdm
import pandas as pd

PARTITIONS = {
    'train' : 0,
    'val'   : 1,
    'test'  : 2,
}

PART_COL = 'partition'

# The aligned CelebA files have `.png` extension.
# However, they have `.jpg` extension in the metadata files (partition/attr).
FILE_EXT  = '.png'
INDEX_EXT = '.jpg'

class CopyWorker:

    def __init__(self, src, dst):
        self._root_src = src
        self._root_dst = dst

    def __call__(self, file_index):
        base, _ext = os.path.splitext(file_index)
        fname      = base + FILE_EXT

        path_src = os.path.join(self._root_src, fname)
        path_dst = os.path.join(self._root_dst, fname)

        shutil.copy(path_src, path_dst)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Prepare CelebA dataset for CycleGAN training'
    )

    parser.add_argument(
        '--list-attr',
        dest     = 'path_attr',
        help     = 'path to `list_attr_celeba.txt`',
        type     = str,
        required = True,
    )

    parser.add_argument(
        '--list-part',
        dest     = 'path_part',
        help     = 'path to `list_eval_partition.txt`',
        type     = str,
        required = True,
    )

    parser.add_argument(
        '--attr',
        dest     = 'attr',
        help     = (
            'attribute to perform a split on (e.g. `Male` or `Eyeglasses`).'
            ' A list of all attributes can be found in the header of'
            ' `list_attr_celeba.txt`.'
        ),
        type     = str,
        required = True,
    )

    parser.add_argument(
        '--separate-val',
        dest     = 'separate_val',
        help     = (
            'by default, validation set will be merged with test set.'
            ' If this flag is set, then the val set will not be merged'
        ),
        action   = 'store_true'
    )

    parser.add_argument(
        '-n', '--workers',
        default = 1,
        dest    = 'workers',
        help    = 'number of parallel workers to use',
        type    = int,
    )

    parser.add_argument(
        'path_celeba',
        metavar  = 'CELEBA',
        help     = (
            'path to the extracted CelebA images `img_align_celeba_png`'
        ),
        type     = str,
    )

    parser.add_argument(
        'outdir',
        metavar  = 'OUTDIR',
        help     = 'output directory',
        type     = str,
    )

    return parser.parse_args()

def load_celeba_attrs(path):
    return pd.read_csv(
        path, sep = r'\s+', skiprows = 1, header = 0, index_col = 0
    )

def load_celeba_partition(path):
    return pd.read_csv(
        path, sep = r'\s+', header = None, names = [ PART_COL, ],
        index_col = 0
    )

def load_celeba_specs(path_attr, path_part):
    df_partition = load_celeba_partition(path_part)
    df_attrs     = load_celeba_attrs(path_attr)

    return df_partition.join(df_attrs)

def collect_celeba_images(root):
    return sorted(os.listdir(root))

def validate_attr(specs, attr):
    attrs = set(specs.columns)
    attrs.remove(PART_COL)

    if attr not in attrs:
        raise ValueError(f"Unknown attribute '{attr}'. Supported: {attrs}")

def validate_images(images, specs):
    assert len(images) == len(specs), (
        f"Number of images {len(images)} does not match the number of"
        f" entries in the file list {len(specs)}."
    )

    for image in images:
        base, ext = os.path.splitext(image)

        assert ext == FILE_EXT, (
            f"Image '{image}' has wrong extension '{ext}'."
            " Expecting '{FILE_EXT}'"
        )

        entry_index = base + INDEX_EXT

        assert entry_index in specs.index, (
            f"Image '{image}' (index '{entry_index}') is not found in the "
            " CelebA file list."
        )

def prepare_single_split(specs, attr, partition):
    part_mask = (specs[PART_COL] == PARTITIONS[partition])

    mask_a = (specs[attr] > 0)  & part_mask
    mask_b = (specs[attr] <= 0) & part_mask

    files_a = specs[mask_a].index.to_list()
    files_b = specs[mask_b].index.to_list()

    return (files_a, files_b)

def prepare_image_split(specs, attr, separate_val):
    train_a, train_b = prepare_single_split(specs, attr, 'train')
    val_a, val_b     = prepare_single_split(specs, attr, 'val')
    test_a, test_b   = prepare_single_split(specs, attr, 'test')

    if separate_val:
        return {
            'trainA' : train_a,
            'trainB' : train_b,
            'valA'   : val_a,
            'valB'   : val_b,
            'testA'  : test_a,
            'testB'  : test_b,
        }

    return {
        'trainA' : train_a,
        'trainB' : train_b,
        'testA'  : test_a + val_a,
        'testB'  : test_b + val_b,
    }

def validate_image_split(specs, split_dict):
    count = sum(len(v) for v in split_dict.values())
    assert len(specs) == count

def prepare_outdir(outdir, split_dict):
    os.makedirs(outdir, exist_ok = True)

    for subdir in split_dict:
        path = os.path.join(outdir, subdir)
        os.mkdir(path)

def split_images(path_celeba, outdir, split_dict, workers):
    for (subdir, files) in split_dict.items():
        curr_outdir = os.path.join(outdir, subdir)

        pbar = tqdm.tqdm(
            desc  = f'Creating {subdir}',
            total = len(files),
            dynamic_ncols = True
        )

        worker = CopyWorker(path_celeba, curr_outdir)

        with multiprocessing.Pool(processes = workers) as pool:
            for _ in pool.imap_unordered(worker, files):
                pbar.update()

        pbar.close()

def main():
    cmdargs = parse_cmdargs()

    if os.path.exists(cmdargs.outdir):
        raise RuntimeError(
            f"Output directory '{cmdargs.outdir}' exists."
            " Refusing to overwrite"
        )

    specs  = load_celeba_specs(cmdargs.path_attr, cmdargs.path_part)
    validate_attr(specs, cmdargs.attr)

    images = collect_celeba_images(cmdargs.path_celeba)
    validate_images(images, specs)

    split_dict = prepare_image_split(specs, cmdargs.attr, cmdargs.separate_val)
    validate_image_split(specs, split_dict)

    prepare_outdir(cmdargs.outdir, split_dict)

    split_images(
        cmdargs.path_celeba, cmdargs.outdir, split_dict, cmdargs.workers
    )

if __name__ == '__main__':
    main()

