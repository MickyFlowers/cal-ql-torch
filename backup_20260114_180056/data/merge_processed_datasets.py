import argparse
import glob
import os
import shutil


def list_hdf5_files(root_path):
    return sorted(glob.glob(os.path.join(root_path, "*.hdf5")))


def ensure_empty_dir(path, overwrite=False):
    if os.path.exists(path):
        existing = list_hdf5_files(path)
        if existing and not overwrite:
            raise RuntimeError(f"Output directory not empty: {path}")
    os.makedirs(path, exist_ok=True)


def copy_file(src, dst, mode):
    if mode == "symlink":
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def merge_datasets(
    old_root, new_root, out_root, mode="copy", old_prefix="old_", new_prefix="new_", overwrite=False
):
    ensure_empty_dir(out_root, overwrite=overwrite)

    old_files = list_hdf5_files(old_root)
    new_files = list_hdf5_files(new_root)
    if not old_files:
        raise RuntimeError(f"No .hdf5 files found in {old_root}")
    if not new_files:
        raise RuntimeError(f"No .hdf5 files found in {new_root}")

    for idx, src in enumerate(old_files):
        dst = os.path.join(out_root, f"{old_prefix}{idx:06d}.hdf5")
        copy_file(src, dst, mode)

    for idx, src in enumerate(new_files):
        dst = os.path.join(out_root, f"{new_prefix}{idx:06d}.hdf5")
        copy_file(src, dst, mode)

    print(f"Merged {len(old_files)} old + {len(new_files)} new episodes into {out_root}")
    print("Note: statistics.yaml/meta.yaml are not copied; they will be rebuilt during training.")


def main():
    parser = argparse.ArgumentParser(description="Merge two processed datasets.")
    parser.add_argument("--old_root", required=True, help="Path to old processed dataset")
    parser.add_argument("--new_root", required=True, help="Path to new processed dataset")
    parser.add_argument("--out_root", required=True, help="Output directory for merged dataset")
    parser.add_argument(
        "--mode",
        default="copy",
        choices=["copy", "symlink", "hardlink"],
        help="How to add files to output directory",
    )
    parser.add_argument("--old_prefix", default="old_", help="Prefix for old dataset files")
    parser.add_argument("--new_prefix", default="new_", help="Prefix for new dataset files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    args = parser.parse_args()

    merge_datasets(
        args.old_root,
        args.new_root,
        args.out_root,
        mode=args.mode,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
