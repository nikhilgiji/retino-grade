import yaml
from pydantic import BaseModel
from src.inference import inference

class ConfigModel(BaseModel):
    train_csv: str
    val_csv: str
    test_csv: str
    train_dir: str
    val_dir: str
    test_dir: str
    output_dir: str
    log_dir: str
    batch_size: int
    img_size: tuple
    seed: int

def main():
    with open('configs/config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    config = ConfigModel(**cfg_dict)
    
    model_path = 'models/best_model_cbam.h5'  # adjust if needed
    
    inference(config, model_path)

if __name__ == "__main__":
    main()