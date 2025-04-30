#  Copyright (c) 2022. \n Created by Hernan Dario Perez
import torch
from model import GenreWithVoiceMutesVAE
from logging import getLogger
logger = getLogger("helpers/GenDensityTempoVAE/modelLoader.py")
logger.setLevel("DEBUG")

def load_GenreWithVoiceMutesVAE_model(model_path, params_dict=None, is_evaluating=True, device=None):
    """ Load a GenreGlobalDensityWithVoiceMutesVAE model from a given path

    Args:
        model_path (str): path to the model
        params_dict (None, dict, or json path): dictionary containing the parameters of the model
        is_evaluating (bool): if True, the model is set to eval mode
        device (None or torch.device): device to load the model to (if cpu, the model is loaded to cpu)

    Returns:
        model (GenreDensityTempoVAE): the loaded model
    """

    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"Model was loaded to cpu!!!")

    if params_dict is None:
        if 'params' in loaded_dict:
            params_dict = loaded_dict['params']
        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        import json
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    model = GenreWithVoiceMutesVAE(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model


if __name__ == "__main__":
    config = {
        'd_model_enc': 128,
        'd_model_dec': 128,
        'embedding_size_src': 3,
        'embedding_size_tgt': 27,
        'nhead_enc': 4,
        'nhead_dec': 8,
        'dim_feedforward_enc': 128,
        'dim_feedforward_dec': 512,
        'num_encoder_layers': 3,
        'num_decoder_layers': 6,
        'dropout': 0.1,
        'latent_dim': 16,
        'max_len_enc': 32,
        'max_len_dec': 32,
        'velocity_dropout': 0.1,
        'offset_dropout': 0.2,
        'n_genres': 12,
        'num_tempo_bins': 3,
        'n_density_bins': 3,
        'device': 'cpu'
    }

    model_ = GenreWithVoiceMutesVAE(config)
    model_.save(save_path="misc/genreDensityTempoVAE.pth")
    load_GenreWithVoiceMutesVAE_model("misc/genreDensityTempoVAE.pth")

