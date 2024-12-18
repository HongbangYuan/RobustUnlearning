from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, try_download_model_from_ms
from .adapter import init_adapter
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from .utils.misc import register_autoclass
from .utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .utils.unsloth import load_unsloth_pretrained_model
from .utils.valuehead import load_valuehead_params
from .lat_model import LATLlamaForCausalLM

import os
import json

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)
from accelerate import infer_auto_device_map, init_empty_weights

class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

def read_config(model_name_or_path):
    config_path = os.path.join(model_name_or_path, "config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)

    if model_args.visual_inputs:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        setattr(processor, "tokenizer", tokenizer)
    else:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable, add_valuehead)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        elif model_args.visual_inputs:
            model = AutoModelForVision2Seq.from_pretrained(**init_kwargs)
        else:
            if finetuning_args.use_lat:
                model_info = read_config(init_kwargs['pretrained_model_name_or_path'])
                try:
                    model_type = model_info["architectures"][0]
                except Exception as e:
                    print(f"Could not find model type using model_info[\"architectures\"][0] in {init_kwargs['pretrained_model_name_or_path']}")
                    raise e

                if model_type == 'LlamaForCausalLM':
                    import transformers
                    version = transformers.__version__
                    if version == '4.44.0':
                        print(f"Transformers version: {version}")
                        print("Load LATLlamaForCausalLM from lat_llama3_new...")
                        from ..model.lat_llama3_new import LATLlamaForCausalLM
                    else:
                        from ..model.lat_model import LATLlamaForCausalLM
                    # with init_empty_weights():
                    #     model = LATLlamaForCausalLM(config)
                    # device_map = infer_auto_device_map(model,no_split_module_classes=["LATLlamaAttention","LATLlamaMLP"])
                    # init_kwargs["device_map"] = device_map
                    # print(f"Setting device map: {device_map}")
                    model = LATLlamaForCausalLM.from_pretrained(**init_kwargs)
                elif model_type == 'Phi3ForCausalLM':
                    from .lat_phi3 import LATPhi3ForCausalLM
                    model = LATPhi3ForCausalLM.from_pretrained(**init_kwargs)
                elif model_type == 'MistralForCausalLM':
                    from .lat_mistral import LATMistralForCausalLM
                    model = LATMistralForCausalLM.from_pretrained(**init_kwargs)
                else:
                    raise f"Model type {model_type} not supported! Please choose LlamaForCausalLM or Phi3ForCausalLM!"
            else:
                model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:d}".format(all_param)
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model