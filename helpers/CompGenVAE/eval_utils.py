#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
from model import ComplexityGenreVAE
from eval.GrooveEvaluator import load_evaluator_template
from eval.UMAP import UMapper

from logging import getLogger
logger = getLogger("helpers.VAE.eval_utils")
logger.setLevel("DEBUG")
from data import Groove2Drum2BarDataset

def generate_umap_for_vae_model_wandb(
        compgen_vae, device, dataset_setting_json_path, subset_name,
        down_sampled_ratio = 0.3):
    """
    Generate the umap for the given model and dataset setting.
    Args:
        :param groove_transformer_vae: The model to be evaluated
        :param device: The device to be used for evaluation
        :param dataset_setting_json_path: The path to the dataset setting json file
        :param subset_name: The name of the subset to be evaluated
        :param collapse_tapped_sequence: Whether to collapse the tapped sequence or not (input will have 1 voice only)

    Returns:
        dictionary ready to be logged by wandb {f"{subset_name}_{umap}": wandb.Html}
    """

    # and model is correct type
    assert isinstance(compgen_vae, ComplexityGenreVAE)

    test_dataset = Groove2Drum2BarDataset(
            dataset_setting_json_path=dataset_setting_json_path,
            subset_tag=subset_name,
            max_len=32,
            tapped_voice_idx=2,
            collapse_tapped_sequence=True,
            down_sampled_ratio=down_sampled_ratio,
            move_all_to_gpu=False,
            use_cached=True
        )

    tags = [hvo_seq.metadata["style_primary"] for hvo_seq in test_dataset.hvo_sequences]

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    latents_z = None
    for batch_ix, batch_in in enumerate(dataloader):
        _, z = compgen_vae.predict(flat_hvo_groove=batch_in[0].to(device))
        if latents_z is None:
            latents_z = z.detach().cpu().numpy()
        else:
            latents_z = np.concatenate((latents_z, z.detach().cpu().numpy()), axis=0)

    try:
        umapper = UMapper(subset_name)
        umapper.fit(latents_z, tags_=tags)
        p = umapper.plot(show_plot=False, prepare_for_wandb=True)
        return {f"{subset_name}_umap": p}

    except Exception as e:
        logger.info("UMAP failed for subset: {}".format(subset_name))
        return None


def get_logging_media_for_vae_model_wandb(
        compgen_vae, device, dataset_setting_json_path, subset_name,
        down_sampled_ratio, collapse_tapped_sequence,
        cached_folder="cached/Evaluators/templates/",
        divide_by_genre=True, **kwargs):
    """
    Prepare the media for logging in wandb. Can be easily used with an evaluator template
    (A template can be created using the code in eval/GrooveEvaluator/templates/main.py)
    :param compgen_vae: The model to be evaluated
    :param device: The device to be used for evaluation
    :param dataset_setting_json_path: The path to the dataset setting json file
    :param subset_name: The name of the subset to be evaluated
    :param down_sampled_ratio: The ratio of the subset to be evaluated
    :param collapse_tapped_sequence: Whether to collapse the tapped sequence or not (input will have 1 voice only)
    :param cached_folder: The folder to be used for caching the evaluator template
    :param divide_by_genre: Whether to divide the subset by genre or not
    :param kwargs:                  additional arguments: need_hit_scores, need_velocity_distributions,
                                    need_offset_distributions, need_rhythmic_distances, need_heatmap
    :return:                        a ready to use dictionary to be logged using wandb.log()
    """

    # and model is correct type
    # assert isinstance(complexity_genre_vae, GrooveTransformerEncoderVAE)

    # Todo: Manually load Evaluator class, with custom dict of params, i.e. Density
    # https://github.com/behzadhaki/GrooveTransformer/blob/main/documentation/chapter3_Evaluator/1_grooveevalbasics.md#22-initialization-

    # load the evaluator template (or create a new one if it does not exist)
    evaluator = load_evaluator_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name=subset_name,
        down_sampled_ratio=down_sampled_ratio,
        cached_folder=cached_folder,
        divide_by_genre=divide_by_genre
    )

    # logger.info("Generating the PianoRolls for subset: {}".format(subset_name))

    # Prepare the flags for require media
    # ----------------------------------
    need_hit_scores = kwargs["need_hit_scores"] if "need_hit_scores" in kwargs.keys() else False
    need_velocity_distributions = kwargs["need_velocity_distributions"] \
        if "need_velocity_distributions" in kwargs.keys() else False
    need_offset_distributions = kwargs["need_offset_distributions"] \
        if "need_offset_distributions" in kwargs.keys() else False
    need_rhythmic_distances = kwargs["need_rhythmic_distances"] \
        if "need_rhythmic_distances" in kwargs.keys() else False
    need_heatmap = kwargs["need_heatmap"] if "need_heatmap" in kwargs.keys() else False
    need_global_features = kwargs["need_global_features"] \
        if "need_global_features" in kwargs.keys() else False
    need_piano_roll = kwargs["need_piano_roll"] if "need_piano_roll" in kwargs.keys() else False
    need_audio = kwargs["need_audio"] if "need_audio" in kwargs.keys() else False
    need_kl_oa = kwargs["need_kl_oa"] if "need_kl_oa" in kwargs.keys() else False

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    hvo_seqs = evaluator.get_ground_truth_hvo_sequences()
    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in hvo_seqs]), dtype=torch.float32).to(
        device)


    in_groove = in_groove.to(device)

    hvos_array, latent_z = compgen_vae.predict(flat_hvo_groove=in_groove)

    evaluator.add_predictions(hvos_array.detach().cpu().numpy())


    # Get the media from the evaluator
    # -------------------------------
    media = evaluator.get_logging_media(
        prepare_for_wandb=True,
        need_hit_scores=need_hit_scores,
        need_velocity_distributions=need_velocity_distributions,
        need_offset_distributions=need_offset_distributions,
        need_rhythmic_distances=need_rhythmic_distances,
        need_heatmap=need_heatmap,
        need_global_features=need_global_features,
        need_piano_roll=need_piano_roll,
        need_audio=need_audio,
        need_kl_oa=need_kl_oa)

    return media


def get_hit_scores_for_vae_model(compgen_vae, device, dataset_setting_json_path, subset_name,
                                 down_sampled_ratio, collapse_tapped_sequence,
                                 cached_folder="cached/GrooveEvaluator/templates/",
                                 divide_by_genre=True):

    # logger.info("Generating the hit scores for subset: {}".format(subset_name))
    # and model is correct type

    assert isinstance(compgen_vae, ComplexityGenreVAE)

    # load the evaluator template (or create a new one if it does not exist)
    evaluator = load_evaluator_template(
        dataset_setting_json_path=dataset_setting_json_path,
        subset_name=subset_name,
        down_sampled_ratio=down_sampled_ratio,
        cached_folder=cached_folder,
        divide_by_genre=divide_by_genre
    )

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    hvo_seqs = evaluator.get_ground_truth_hvo_sequences()
    in_groove = torch.tensor(
        np.array([hvo_seq.flatten_voices(reduce_dim=collapse_tapped_sequence)
                  for hvo_seq in hvo_seqs]), dtype=torch.float32).to(
        device)


    predictions = []


    # batchify the input
    compgen_vae.eval()
    with torch.no_grad():
        for batch_ix, batch_in in enumerate(torch.split(in_groove, 32)):
            in_ = batch_in.to(device)
            hvos_array, latent_z = compgen_vae.predict(flat_hvo_groove=in_)
            predictions.append(hvos_array.detach().cpu().numpy())

    evaluator.add_predictions(np.concatenate(predictions))

    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"Hit_Scores/{key}_mean_{subset_name}".replace(" ", "_").replace("-", "_"): float(value['mean']) for key, value
                  in sorted(hit_dict.items())}

    score_dict.update({f"Hit_Scores/{key}_std_{subset_name}".replace(" ", "_").replace("-", "_"): float(value['std']) for key, value
                  in sorted(hit_dict.items())})

    return score_dict
