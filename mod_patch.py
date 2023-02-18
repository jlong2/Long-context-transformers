#from clm_dataset import CLMDataset, PREFIX_TOKEN_TYPE_ID, PAD_TOKEN_TYPE_ID, IGNORE_TOKEN_ID
PREFIX_TOKEN_TYPE_ID = 0
TARGETS_TOKEN_TYPE_ID = 1
PAD_TOKEN_TYPE_ID = 2
SEPARATOR_TOKEN_TYPE_ID = 3
IGNORE_TOKEN_ID = -100
from typing import Optional,Tuple,Union
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

def patch_forward(
    self,
    input_ids = None,
    attention_mask = None,
    inputs_embeds = None,
    head_mask = None,
    past_key_values = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
):
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
        only required when the model is used as a decoder in a Sequence to Sequence model.
        Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
        `past_key_values` input) to speed up sequential decoding.
        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
        `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
        ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
    Returns:
    Example:
    ```python
    >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
    >>> import torch
    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config.is_decoder = True
    >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)
    >>> prediction_logits = outputs.logits
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.gpt_neox(
        input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    lm_logits = self.embed_out(hidden_states)

    lm_loss = None
    if labels is not None:
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((lm_loss,) + output) if lm_loss is not None else output

    return CausalLMOutputWithPast(
        loss=lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

#patch on
#https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2610-L2640
def patch_compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.
    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    #import pdb; pdb.set_trace()
    #outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    #input_ids, position_ids, token_type_ids = prepare_inputs(args, batch, device)
    #outputs = model(input_ids=input_ids, position_ids=position_ids, labels=input_ids)
    input_ids = inputs.pop("input_ids")
    token_type_ids = inputs.pop("token_type_ids")
    outputs = model(input_ids=input_ids, labels=input_ids)
    flattened_loss = outputs[0] if isinstance(outputs,tuple) else outputs.loss
    loss_scale = loss_scale_of_crossentropy(targets=input_ids[..., 1:],
                                                ignore_index=IGNORE_TOKEN_ID,
                                                target_token_type_ids=token_type_ids[..., 1:],
                                                prompt_loss_weight=0.5
    )
    loss = (flattened_loss * loss_scale).sum()
    #import pdb; pdb.set_trace()
    print(loss_scale.shape, flush=True)
    print("flattened_loss",flattened_loss.shape, flush=True)
    print("loss 1", loss, flush=True)
    

    if labels is not None:
        if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
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
        #we already compute loss above #loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    print("loss 2", loss, flush=True)
    return (loss, outputs) if return_outputs else loss

def prepare_inputs(args, batch, device):
    input_ids = batch[0].to(device).to(torch.int64)
    batch_size = input_ids.shape[0]
    
    position_ids = torch.arange(args.max_seq_length)
    position_ids = position_ids.unsqueeze(0).expand(input_ids.shape)
    position_ids = position_ids.to(device)

    token_type_ids = batch[1]

    return input_ids, position_ids, token_type_ids

def loss_scale_of_crossentropy(
        targets: torch.Tensor,
        ignore_index: int,
        target_token_type_ids: torch.Tensor,
        prompt_loss_weight: float = 0.5):
    assert targets.shape == target_token_type_ids.shape
    grad_scale_not_ignored = ~targets.eq(ignore_index)
    grad_scale_not_ignored[target_token_type_ids.eq(PAD_TOKEN_TYPE_ID)] = False

    grad_scale = grad_scale_not_ignored.float()
    grad_scale[target_token_type_ids.eq(PREFIX_TOKEN_TYPE_ID)] *= prompt_loss_weight
    grad_scale /= torch.sum(grad_scale)
    return grad_scale.flatten()
    