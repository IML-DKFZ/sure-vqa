import torch.cuda

from med_vlm_robustness.datamodule import get_datamodule
from med_vlm_robustness.model import LLaVA_Med
from pytorch_lightning import Trainer


def main():
    llava = LLaVA_Med()
    dm = get_datamodule("slake_test_ood_modality_X-Ray", batch_size=1)
    dm.setup()

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(llava, datamodule=dm)

if __name__ == "__main__":
    main()