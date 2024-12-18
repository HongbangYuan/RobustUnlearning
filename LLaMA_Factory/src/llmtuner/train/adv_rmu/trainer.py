from types import MethodType
from typing import TYPE_CHECKING, Optional

from transformers import Trainer
import torch.nn.functional as F
from torch import nn
import torch
from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler
from trl.trainer.sft_trainer import unwrap_model, PeftModel, is_peft_available
# from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
# from transformers.trainer import is_sagemaker_mp_enabled,is_xpu_available,is_mlu_available,is_npu_available,is_torch_version,is_mps_available
# from transformers.trainer import OptimizerNames
from transformers.trainer import *
# from .......utils.nethook import TraceDict, layername
from ..utils import TraceDict, layername
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class AdvRMUTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(self, ref_model, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        self.rmu_layer = finetuning_args.rmu_layer
        self.rmu_layer_name = layername(self.model,self.rmu_layer)
        self.steering_coeff = finetuning_args.rmu_steering_coeff
        self.alpha = finetuning_args.rmu_alpha

        self.perturb_layer = finetuning_args.adv_rmu_perturb_layer
        self.epsilon = finetuning_args.adv_npo_epsilon
        self.steps = finetuning_args.adv_npo_steps
        self.norm_type = finetuning_args.adv_npo_norm_type
        self.random_init = finetuning_args.adv_npo_random_init
        self.std_normalization = finetuning_args.adv_npo_std_normalization
        self.keep_in_eval = finetuning_args.adv_npo_keep_in_eval
        self.perturb_target = finetuning_args.adv_npo_perturb_target

        if self.epsilon == 0.0:
            raise ValueError(f"Self epsilon is set to {self.epsilon} in adv process!")


        self.ref_model = ref_model

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)


    def get_perturbation(self,model,inputs,return_outputs=False):
        if "prompt_0" in inputs:
            forget_inputs, retain_inputs = inputs["prompt_0"], inputs["prompt_1"]
            # get the forget activations after the adversary process
            perturbation = self.compute_adv_loss(model, forget_inputs)
            return perturbation
        raise NotImplementedError("Retain set is not specified! Got only forget set!")


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        perturbation = self.get_perturbation(model,inputs)
        outer_steps = 4
        total_loss = None
        for _ in range(outer_steps):
            with self.compute_loss_context_manager():
                loss = self.compute_loss_with_perturbation(model, inputs,perturbation)
                # if (
                #     self.args.torch_empty_cache_steps is not None
                #     and self.state.global_step % self.args.torch_empty_cache_steps == 0
                # ):
                #     if is_xpu_available():
                #         torch.xpu.empty_cache()
                #     elif is_mlu_available():
                #         torch.mlu.empty_cache()
                #     elif is_npu_available():
                #         torch.npu.empty_cache()
                #     elif is_torch_version(">=", "2.0") and is_mps_available():
                #         torch.mps.empty_cache()
                #     else:
                #         torch.cuda.empty_cache()

                kwargs = {}

                # # For LOMO optimizers you need to explicitly use the learnign rate
                # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                #     kwargs["learning_rate"] = self._get_learning_rate()

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss, **kwargs)


                if total_loss is None:
                    total_loss = loss.detach()
                else:
                    total_loss += loss.detach().to(total_loss.device)  # 累加张量 loss

        del inputs
        total_loss = total_loss / outer_steps
        return total_loss.detach() / self.args.gradient_accumulation_steps


    def compute_loss_with_perturbation(self,model,inputs,perturbation):
        forget_inputs,retain_inputs = inputs["prompt_0"],inputs["prompt_1"]
        forget_input_ids, forget_labels, forget_attention_mask = forget_inputs['input_ids'], forget_inputs['labels'], forget_inputs['attention_mask']
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs["input_ids"],retain_inputs["labels"],retain_inputs["attention_mask"]

        # *************** unlearning loss****************
        # ***************  forget loss
        with TraceDict(model, [self.rmu_layer_name], retain_grad=True) as ret:
            outputs = model(**forget_inputs, perturb_layer=self.perturb_layer, perturbation=perturbation, perturb_target=self.perturb_target)
        # Note: the output of the lat-decode layer is (outputs,latents). That's why we add an additional 0 index comparing with the rmu trainer.
        forget_activations = ret[self.rmu_layer_name].output[0][0] # (batch_size,seq_len,hidden_size)

        seq_length = forget_activations.shape[1]
        batch_size = forget_activations.shape[0]

        random_vector = torch.rand(batch_size,seq_length,self.model.config.hidden_size,dtype=forget_activations.dtype,device=forget_activations.device)
        control_vec = random_vector / torch.norm(random_vector) * self.steering_coeff
        forget_loss = torch.nn.functional.mse_loss(
            forget_activations,control_vec
        ).to(torch.device("cuda:0"))
        # return forget_loss
        # ***************  retain loss
        with TraceDict(self.ref_model,[self.rmu_layer_name],retain_grad=True) as ret:
            retain_outputs = self.ref_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask,output_hidden_states=True)
        # Note: an additional 0 index!
        frozen_retain_activations = ret[self.rmu_layer_name].output[0][0].to(forget_activations.dtype).to(forget_activations.device) # (batch_size,seq_len,hidden_size)
        # frozen_retain_activations = retain_outputs["hidden_states"][self.rmu_layer].to(forget_activations.dtype)

        with TraceDict(model,[self.rmu_layer_name],retain_grad=True) as ret:
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask,output_hidden_states=True)
        # Note: an additional 0 index!
        updated_retain_activations = ret[self.rmu_layer_name].output[0][0].to(forget_activations.device) # (batch_size,seq_len,hidden_size)
        # updated_retain_activations = retain_outputs["hidden_states"][self.rmu_layer]

        retain_loss = torch.nn.functional.mse_loss(
            updated_retain_activations,frozen_retain_activations
        ).to(torch.device("cuda:0"))
        # print(f"Forget loss device:{forget_loss.device}, Retain Loss device:{retain_loss.device}")

        total_loss = forget_loss + retain_loss * self.alpha
        return total_loss
        # return retain_loss

    def compute_adv_loss(self,model,inputs):
        # return: (forget_loss, forget activations) Note: they are all computed after the adv process

        if self.keep_in_eval:
            model.eval()


        perturbation = self.adv_attack(model, **inputs)
        # with TraceDict(model, [self.rmu_layer_name], retain_grad=True) as ret:
        #     outputs = model(**inputs, perturb_layer=self.perturb_layer, perturbation=perturbation, perturb_target=self.perturb_target)
        # Note: the output of the lat-decode layer is (outputs,latents). That's why we add an additional 0 index comparing with the rmu trainer.
        # forget_activations = ret[self.rmu_layer_name].output[0][0] # (batch_size,seq_len,hidden_size)

        return perturbation


    def adv_attack(self, model, **inputs):
        '''
        :param model: the model to attack
        :param inputs:
        :return: an adversarial perturbation to the embeddings or latents
        '''
        print("Start the adv attack process...")
        if self.norm_type != 'l2':
            raise NotImplementedError

        model_training = model.training
        model.eval()  # attack the model in eval mode
        model.zero_grad()

        input = {}
        for key in inputs.keys():
            input[key] = inputs[key].detach().clone()

        model_latents_orig, model_output_orig = model(**inputs, perturb_layer=self.perturb_layer,
                                                      get_latents=True, perturb_target=self.perturb_target)
        min_act = torch.min(model_latents_orig)  # min and max acts will be used for clipping
        max_act = torch.max(model_latents_orig)  # min and max acts will be used for clipping
        batch_size = model_latents_orig.shape[0]

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.random_init or self.steps == 0:  # steps == 0 means that it'll just be a random latent perturbation
            current_perturbation = torch.empty_like(model_latents_orig).normal_().to(model_latents_orig.device)
            d_flat = torch.reshape(current_perturbation, (batch_size, -1))
            if len(current_perturbation.shape) == 4:  # Q, K, or V perturbation
                n = d_flat.norm(p=2, dim=1).view(batch_size, 1, 1, 1)
            else: # residual perturbation
                n = d_flat.norm(p=2, dim=1).view(batch_size, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            current_perturbation *= ((r / n) * self.epsilon)
        else:
            current_perturbation = torch.zeros_like(model_latents_orig).to(model_latents_orig.device)

        stepsize = self.epsilon / max(1, self.steps)
        print(f"Caculated stepsize: {stepsize}")
        if self.std_normalization:
            activation_std = torch.std(model_latents_orig, dim=0, keepdim=False)
            mean_std = activation_std.mean()
            normalization = (activation_std / mean_std) + (torch.mean(torch.abs(model_latents_orig)) / 10)
        else:
            normalization = 1.0

        def project_scale_clip(_current_perturbation):
            perturbation_norms = torch.norm(torch.reshape(_current_perturbation, (batch_size, -1)), p=2, dim=1)
            factor = self.epsilon / perturbation_norms
            factor = torch.min(factor, torch.ones_like(perturbation_norms))
            if len(_current_perturbation.shape) == 4:  # Q, K, or V perturbation
                _current_perturbation = _current_perturbation * factor.view(-1, 1, 1, 1)
            else:  # residual perturbation
                _current_perturbation = _current_perturbation * factor.view(-1, 1, 1)
            adv_latents = torch.clamp(model_latents_orig + (_current_perturbation * normalization),
                                      min=min_act,
                                      max=max_act).detach()
            _current_perturbation = ((adv_latents - model_latents_orig) / normalization).detach()
            return _current_perturbation

        current_perturbation = project_scale_clip(current_perturbation)

        # if not doing random latent perturbations
        for step in range(self.steps):
            print(f"--------------Adv Attack Step {step}-----------")
            model.zero_grad()

            # Get loss
            current_perturbation.requires_grad = True
            model_outputs_pert = model(**inputs, perturb_layer=self.perturb_layer,
                                       perturbation=current_perturbation * normalization,
                                       perturb_target=self.perturb_target)
            if labels is not None:
                unwrapped_model = unwrap_model(model)
                if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(model_outputs_pert, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(model_outputs_pert, labels)
            else:
                if isinstance(model_outputs_pert, dict) and "loss" not in model_outputs_pert:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(model_outputs_pert.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = model_outputs_pert["loss"] if isinstance(model_outputs_pert, dict) else model_outputs_pert[0]

            # print(loss.item(), step)  # you can uncomment this to check that the loss goes up

            # backprop
            loss.backward(retain_graph=True)
            print(f"Loss: {loss.item()}")

            # Get new perturbation
            grad = current_perturbation.grad
            # print(f"Gradient: {grad}")

            # modify grad as needed
            grad_norms = torch.norm(torch.reshape(grad, (batch_size, -1)), p=2, dim=1) + 1e-6
            if len(grad.shape) == 4:  # Q, K, or V perturbation
                grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            else:  # residual perturbation
                grad = grad / grad_norms.view(batch_size, 1, 1)
            # print(f"Gradient after manual normalization:{grad}")

            # update perturbation
            # current_perturbation = current_perturbation.detach() + stepsize * grad
            # We want to train the model to unlearn the biographies, so the noise should let the model memorize the biographies
            # Therefore, we want to calculate perturbations that have smaller loss
            current_perturbation = current_perturbation.detach() - stepsize * grad # gradient descent, smaller loss on positive examples!

            # project, scale clip
            current_perturbation = project_scale_clip(current_perturbation)
        print("Finished searching for current perturbation!")
        model.zero_grad()
        if model_training and (not self.keep_in_eval):
            model.train()  # put back in train mode

        return current_perturbation * normalization