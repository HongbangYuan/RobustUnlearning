from types import MethodType
from typing import TYPE_CHECKING, Optional
from torch import nn
import torch
from transformers import Trainer
import torch.nn.functional as F
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from trl.trainer.sft_trainer import unwrap_model, PeftModel, is_peft_available

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class LATSFTTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self,
        ref_model,
        finetuning_args: "FinetuningArguments",
        **kwargs
    ) -> None:

        self.perturb_layer = finetuning_args.adv_npo_perturb_layer
        self.epsilon = finetuning_args.adv_npo_epsilon
        self.steps = finetuning_args.adv_npo_steps
        self.norm_type = finetuning_args.adv_npo_norm_type
        self.random_init = finetuning_args.adv_npo_random_init
        self.std_normalization = finetuning_args.adv_npo_std_normalization
        self.keep_in_eval = finetuning_args.adv_npo_keep_in_eval
        self.perturb_target = finetuning_args.adv_npo_perturb_target

        self.npo_coeff = finetuning_args.npo_coeff
        self.grad_diff_coeff = finetuning_args.grad_diff
        self.KL_coeff = finetuning_args.KL_coeff
        self.loss_type = finetuning_args.npo_loss

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.ref_model = ref_model
        self.beta = finetuning_args.dpo_beta
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

    def get_batch_loss(self, output, labels):
        shifted_labels = labels[..., 1:].contiguous()
        output = output[..., :-1, :].contiguous()

        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # get the sum loss for each sequence in a batch
        loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

        return loss

    def compute_adv_loss(self,model,inputs,return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.keep_in_eval:
            model.eval()
        if self.epsilon == 0.0 or (not self.is_in_train):  # normal training or eval
            outputs = model(**inputs)
        else:  # [latent] adv training
            perturbation = self.adv_attack(model, **inputs)
            outputs = model(**inputs, perturb_layer=self.perturb_layer, perturbation=perturbation, perturb_target=self.perturb_target)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

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

    def compute_loss_with_retain(self,model,inputs,return_outputs=False):
        forget_inputs,retain_inputs = inputs["prompt_0"],inputs["prompt_1"]
        input_ids, labels, attention_mask = forget_inputs['input_ids'], forget_inputs['labels'], forget_inputs['attention_mask']
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs["input_ids"],retain_inputs["labels"],retain_inputs["attention_mask"]

        if self.loss_type == 'npo':
            loss, outputs = self.compute_adv_loss(model, forget_inputs, True)
            forget_loss_current = self.get_batch_loss(outputs.logits, forget_inputs['labels'])

            with torch.no_grad():
                forget_outputs_oracle = self.ref_model(forget_inputs['input_ids'], labels=forget_inputs['labels'],
                                                       attention_mask=forget_inputs['attention_mask'])
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, forget_inputs['labels'])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
            loss = forget_loss
        elif self.loss_type == 'npo_grad_diff':
            loss, outputs = self.compute_adv_loss(model, forget_inputs, True)
            forget_loss_current = self.get_batch_loss(outputs.logits, forget_inputs['labels'])
            with torch.no_grad():
                forget_outputs_oracle = self.ref_model(forget_inputs['input_ids'], labels=forget_inputs['labels'],
                                                       attention_mask=forget_inputs['attention_mask'])
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, forget_inputs['labels'])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs["input_ids"],retain_inputs["labels"],retain_inputs["attention_mask"]
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
        elif self.loss_type == 'npo_KL':
            loss,outputs = self.compute_adv_loss(model,forget_inputs,True)
            forget_loss_current = self.get_batch_loss(outputs.logits, forget_inputs['labels'])
            with torch.no_grad():
                forget_outputs_oracle = self.ref_model(forget_inputs['input_ids'], labels=forget_inputs['labels'], attention_mask=forget_inputs['attention_mask'])
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, forget_inputs['labels'])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            with torch.no_grad():
                retain_outputs = self.ref_model(retain_input_ids, labels=retain_labels,
                                                   attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            loss = self.npo_coeff * forget_loss + self.KL_coeff * retain_loss
        else:
            raise NotImplementedError(f"{self.loss_type} loss type not implemented!")

        return (loss, outputs) if return_outputs else loss


    def compute_loss(self, model, inputs, return_outputs=False):
        # loss, outputs = super().compute_loss(model, inputs, True)
        if "prompt_0" in inputs:
            return self.compute_loss_with_retain(model,inputs,return_outputs)
        loss,outputs = self.compute_adv_loss(model,inputs,True)
        forget_loss_current = self.get_batch_loss(outputs.logits, inputs['labels'])
        with torch.no_grad():
            forget_outputs_oracle = self.ref_model(inputs['input_ids'], labels=inputs['labels'], attention_mask=inputs['attention_mask'])
            forget_logits_oracle = forget_outputs_oracle.logits
            forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, inputs['labels'])
        neg_log_ratios = forget_loss_current - forget_loss_oracle
        loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
        return (loss, outputs) if return_outputs else loss
