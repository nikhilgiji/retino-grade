import os
import logging
from tqdm import tqdm
import tensorflow as tf

from src.data_loader import DataLoader, DataLoaderConfig
from src.model import build_model_with_cbam
from src.callbacks import MacroF1Callback
from src.utils import setup_logger, create_dirs

def train(config):
    # Setup directories and logger
    create_dirs([config.model_dir, config.output_dir, config.log_dir])
    logger = setup_logger("train_logger", os.path.join(config.log_dir, "train.log"))

    # Data
    data_loader = DataLoader(config)
    train_generator = data_loader.get_train_generator()
    val_generator = data_loader.get_val_generator()

    # Build model
    model = build_model_with_cbam(config.img_size, n_classes=5)
    logger.info("Model built successfully")

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.model_dir, "best_model_cbam.h5"),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config.log_dir, 'tensorboard'))

    macro_f1_cb = MacroF1Callback(val_generator)

    callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb, tensorboard_cb, macro_f1_cb]

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Training completed")

    return model, history