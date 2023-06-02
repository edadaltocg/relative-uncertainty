import os
import time
import torch
from tools import doctor_tools
from tools import data_tools, ml_tools


def compute_doctor_scores_for_magnitude_and_temperature(temperature,
                                                        magnitude,
                                                        logger,
                                                        new_match_ts_logits,
                                                        new_mismatch_logits,
                                                        dest_folder) -> None:
    magnitude_folder = '/'.join((dest_folder, f"magnitude_{magnitude}"))
    if not os.path.exists(magnitude_folder):
        os.mkdir(magnitude_folder)
    temperature_folder = '/'.join((magnitude_folder,
                                   f"/temperature_{temperature}"))
    if not os.path.exists(temperature_folder):
        os.mkdir(temperature_folder)
    logger.info(f"Magnitude {magnitude}\tTemperature: {temperature}")
    new_match_ts_doctor_scores = doctor_tools.doctor(
        logits=new_match_ts_logits, temperature=temperature)
    new_mismatch_ts_doctor_scores = doctor_tools.doctor(
        logits=new_mismatch_logits, temperature=temperature)

    # save doctor scores
    torch.save(new_match_ts_doctor_scores,
               f"{temperature_folder}/match_ts_doctor_scores.pt")
    torch.save(new_mismatch_ts_doctor_scores,
               f"{temperature_folder}/mismatch_doctor_scores.pt")
