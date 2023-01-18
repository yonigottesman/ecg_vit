import argparse
from pathlib import Path

import tensorflow as tf

from data import gen_datasets, gen_df
from vit import VisionTransformer


def train(database):

    df = gen_df(Path(database))
    train_ds, val_ds = gen_datasets(df, 64)

    epochs = 10
    vit = VisionTransformer(
        patch_size=20,
        hidden_size=768,
        depth=6,
        num_heads=6,
        mlp_dim=256,
        num_classes=len(df["y"].values[0]),
        sd_survival_probability=0.9,
    )

    optimizer = tf.keras.optimizers.Adam(0.0001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.AUC(from_logits=True, name="roc_auc")]
    vit.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            "/data/vit_best/", monitor="val_roc_auc", save_best_only=True, save_weights_only=True
        )
    ]

    vit.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", help="root folder of database")

    args = parser.parse_args()
    train(args.database)
