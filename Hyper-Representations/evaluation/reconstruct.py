import os; os.chdir("../method")
import sys; sys.path.append("../method")
import json
import numpy as np
from pathlib import Path
from src.ghrp.model_definitions.def_simclr_ae_module import SimCLRAEModule
import torch

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    experiment_path = Path("./data/hyper_representations/svhn")
    ae_config = json.load((experiment_path / "config_ae.json").open("r"))
    ae_config.update({"device": "cuda", "model::type": "transformer"})
    ae_checkpoint = torch.load(experiment_path / "checkpoint_ae.pt", map_location="cpu", weights_only=False)
    AE = SimCLRAEModule(ae_config)
    AE.model.load_state_dict(ae_checkpoint)
    AE.model.eval()

    training_weights = torch.load(experiment_path / "dataset.pt", weights_only=False)["trainset"].__get_weights__()
    training_weights = training_weights.to("cuda")
    batch_size = 256
    weight_vector = []
    for i in range(0, len(training_weights), batch_size):
        batch_ckpts = training_weights[i:i+batch_size]
        with torch.no_grad():
            batch_weight_vector = AE.forward_decoder(AE.forward_encoder(batch_ckpts)).cpu().detach()
        weight_vector.append(batch_weight_vector)
        del batch_ckpts, batch_weight_vector
        torch.cuda.empty_cache()
    weight_vector = torch.cat(weight_vector, dim=0)
    np.save("../data/reconstructed_weights.npy", weight_vector)