import yaml
from pydantic import BaseModel
from src.train import train

class Config(BaseModel):
    seed: int
    img_size: tuple
    batch_size: int
    epochs: int
    train_csv: str
    val_csv: str
    test_csv: str
    train_dir: str
    val_dir: str
    test_dir: str
    model_dir: str
    output_dir: str
    log_dir: str
    learning_rate: float

def main():
    with open("configs/config.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    config = Config(**cfg_dict)
    
    train(config)

if __name__ == "__main__":
    main()