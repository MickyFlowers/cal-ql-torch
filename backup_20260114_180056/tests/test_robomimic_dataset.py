import h5py
import hydra
import torch

from data.dataset import RoboMimicDataset


@hydra.main(config_path="../config/dataset", config_name="robomimic_dataset")
def main(config):
    dataset = RoboMimicDataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    for i, batch in enumerate(dataloader):
        print(batch["action"].min())


if __name__ == "__main__":
    main()
