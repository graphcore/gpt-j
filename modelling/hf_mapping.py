# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict
import numpy as np

from transformers.models.gptj import GPTJModel as HFModel
from transformers.models.gptj import GPTJForCausalLM as HFLMHeadModel
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTJConfigHF

import popxl
from popxl_addons import TaskSession

from config import GPTJConfig
from modelling.gptj_model import GPTJModelTP
from modelling.gptj_lm import GPTJLMHeadLossAndGradTP, GPTJLMHeadModelTP


def hf_mapping_lm_tp(
    config: GPTJConfig, session: TaskSession, pretrained: HFLMHeadModel
) -> Dict[popxl.Tensor, np.ndarray]:
    load_to = session.state
    if "fwd" in session.state:
        load_to = session.state.fwd
    weights = GPTJLMHeadModelTP.hf_mapping(config, load_to, pretrained)
    return weights


def hf_mapping_TP(config: GPTJConfig, session: TaskSession, pretrained: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
    load_to = session.state
    if "fwd" in session.state:
        load_to = session.state.fwd
    weights = GPTJModelTP.hf_mapping(config, load_to, pretrained)
    return weights


def load_lm_to_hf(session: TaskSession, hf_model: HFLMHeadModel) -> HFLMHeadModel:
    weights = session.get_named_tensors_data()
    if "fwd" in weights:
        weights = weights.fwd
    state_dict = GPTJLMHeadModelTP.to_hf(weights, hf_model)
    # check only missing keys are mask-related keys
    hf_state_keys = hf_model.state_dict().keys()
    popxl_keys = state_dict.keys()

    def should_check(k: str):
        return "attn.bias" not in k and "attn.masked_bias" not in k

    for k in hf_state_keys:
        if should_check(k) and k not in popxl_keys:
            raise KeyError(f"key {k} not found in session state")

    hf_model.load_state_dict(state_dict, strict=False)
    return hf_model


def load_to_hf(session: TaskSession, hf_model: HFModel) -> HFModel:
    weights = session.get_named_tensors_data()
    if "fwd" in weights:
        weights = weights.fwd

    state_dict = GPTJModelTP.to_hf(weights, hf_model)
    # check only missing keys are mask-related keys
    hf_state_keys = hf_model.state_dict().keys()
    popxl_keys = state_dict.keys()

    def should_check(k: str):
        return "attn.bias" not in k and "attn.masked_bias" not in k

    for k in hf_state_keys:
        if should_check(k) and k not in popxl_keys:
            raise KeyError(f"key {k} not found in session state")

    hf_model.load_state_dict(state_dict, strict=False)
    return hf_model
