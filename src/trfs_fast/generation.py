# coding=utf-8
# Copyright 2022 HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import inspect
from typing import Optional, TYPE_CHECKING, Dict, Any

from transformers.utils import logging, ModelOutput
from transformers import GenerationConfig

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

class GenerationPrefill:
    @torch.no_grad()
    def generate_minimal(
        self,
        min_new_tokens: int,
        max_new_tokens: int,
        inputs: Optional[torch.Tensor] = None,
        streamer: Optional["BaseStreamer"] = None,
        cache_length: Optional[int] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        r"""

        Generates sequences of token ids for models with a language modeling head with greedy search.

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        generation_config = GenerationConfig.from_model_config(self.config)

        self._validate_model_kwargs(model_kwargs.copy())

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            raise NotImplementedError("encoder-decoder not supported yet")
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        batch_size, context_length = input_ids.shape
        cache_length = cache_length if cache_length is not None else max_new_tokens

        model_kwargs["valid_past_index"] = torch.tensor(0).to(input_ids.device)
        model_kwargs["past_key_values"] = self.get_empty_kv_cache(batch_size=batch_size, cache_length=cache_length, device=input_ids.device, dtype=self.dtype)
        model_kwargs["attention_mask"] = self.get_preallocated_attention_mask(attention_mask=model_kwargs["attention_mask"], batch_size=batch_size, cache_length=cache_length, device=input_ids.device, context_length=context_length)

        # 11. run greedy search
        return self.greedy_search_minimal(
            input_ids,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            streamer=streamer,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )

    def greedy_search_minimal(
        self,
        input_ids: torch.LongTensor,
        pad_token_id: int,
        eos_token_id: int,
        min_new_tokens: int,
        max_new_tokens: int,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            max_new_tokens (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`):
                The id of the *padding* token.
            eos_token_id (`int`):
                The id of the *end-of-sequence* token.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device).unsqueeze(1)
        n_eos_tokens = eos_token_id_tensor.shape[0]

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        counter = 0
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            counter += 1

            # argmax
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.__update_model_kwargs_for_generation(
                outputs, model_kwargs, model_inputs
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(n_eos_tokens, 1).ne(eos_token_id_tensor).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0 and counter >= min_new_tokens:
                    break

            # stop if we exceed the maximum length
            if counter >= max_new_tokens:
                break

        if streamer is not None:
            streamer.end()

        return input_ids

    def __update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        model_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        model_kwargs["valid_past_index"] += outputs.logits.shape[1]

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update attention mask
        """
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        """
        position_ids = model_inputs["position_ids"]
        if position_ids.shape[1] > 1:
            model_kwargs["position_ids"] = position_ids[:, -1:] + 1
        else:
            model_kwargs["position_ids"] = position_ids + 1

        # NOTE: token_type_ids is not used by llama so we don't care about this one for now
        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        return model_kwargs
