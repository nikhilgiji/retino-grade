import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
from pydantic import BaseModel

class DataLoaderConfig(BaseModel):
    train_csv: str
    val_csv: str
    test_csv: str
    train_dir: str
    val_dir: str
    test_dir: str
    img_size: Tuple[int, int]
    batch_size: int
    seed: int

class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.train_df = pd.read_csv(config.train_csv)
        self.val_df = pd.read_csv(config.val_csv)
        self.test_df = pd.read_csv(config.test_csv)

        # Ensure labels are strings for flow_from_dataframe with categorical class_mode
        self.train_df['diagnosis'] = self.train_df['diagnosis'].astype(str)
        self.val_df['diagnosis'] = self.val_df['diagnosis'].astype(str)
        if 'diagnosis' in self.test_df.columns:
            self.test_df['diagnosis'] = self.test_df['diagnosis'].astype(str)

        self._add_file_paths()

    def _add_file_paths(self):
        self.train_df['file_path'] = self.train_df['id_code'].apply(lambda x: os.path.join(self.config.train_dir, f"{x}.png"))
        self.val_df['file_path'] = self.val_df['id_code'].apply(lambda x: os.path.join(self.config.val_dir, f"{x}.png"))
        self.test_df['file_path'] = self.test_df['id_code'].apply(lambda x: os.path.join(self.config.test_dir, f"{x}.png"))

    def get_train_generator(self) -> ImageDataGenerator:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            zoom_range=0.2,
            rotation_range=20,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        return train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            x_col='file_path',
            y_col='diagnosis',
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=self.config.seed
        )

    def get_val_generator(self) -> ImageDataGenerator:
        val_datagen = ImageDataGenerator(rescale=1./255)
        return val_datagen.flow_from_dataframe(
            dataframe=self.val_df,
            x_col='file_path',
            y_col='diagnosis',
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def get_test_generator(self) -> ImageDataGenerator:
        test_datagen = ImageDataGenerator(rescale=1./255)
        return test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            x_col='file_path',
            y_col='diagnosis' if 'diagnosis' in self.test_df.columns else None,
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical' if 'diagnosis' in self.test_df.columns else None,
            shuffle=False
        )