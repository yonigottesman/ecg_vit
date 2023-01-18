from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from tqdm import tqdm


def read_record(path):
    record = wfdb.rdrecord(path.decode("utf-8"))
    return record.p_signal.astype(np.float32)


def ds_base(df, shuffle, bs):
    ds = tf.data.Dataset.from_tensor_slices((df["file"], list(df["y"])))
    if shuffle:
        ds = ds.shuffle(len(df))
    ds = ds.map(
        lambda x, y: (tf.numpy_function(read_record, inp=[x], Tout=tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(lambda x, y: (tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), y))  # replace nan with zero
    ds = ds.map(lambda x, y: (tf.ensure_shape(x, [5000, 12]), y))
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def gen_datasets(df: pd.DataFrame, bs):
    train_ds = ds_base(df[~df["test"]], True, bs)
    val_ds = ds_base(df[df["test"]], False, bs)
    return train_ds, val_ds


def gen_df(database_root: Path, test_ratio=0.2):
    labels = [59931005, 164873001]
    labels_index = {snomed: i for i, snomed in enumerate(labels)}
    records = []
    failed = []
    for i in tqdm(list(database_root.glob("**/*.hea"))):
        file = i.with_suffix("").as_posix()
        try:
            record = wfdb.rdrecord(file)
        except Exception:
            failed.append(file)
            continue

        metadata = dict([i.split(": ") for i in record.comments])
        y = np.zeros(len(labels_index))
        if "Dx" in metadata:
            snomeds = map(int, metadata["Dx"].split(","))
            indices = [labels_index[i] for i in snomeds if i in labels_index]
            y[indices] = 1
        records.append({"file": file, "y": y})

    df = pd.DataFrame(records)
    df = pd.concat(
        [df[df["y"].apply(lambda y: np.sum(y) == 0)].sample(20000), df[df["y"].apply(lambda y: np.sum(y) != 0)]]
    )
    df["test"] = df["file"].apply(lambda file: crc32(bytes(file, "utf-8")) < test_ratio * 2**32)
    return df
