# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ConceptSeg-R1 model based on Qwen2-VL."""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, \
    Qwen2_5_VLCausalLMOutputWithPast
from transformers.utils import logging
from open_r1.mymodels.qwen2_vl import Qwen2VLVisionConnectorSimple

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logger = logging.get_logger(__name__)
local_rank = int(os.getenv("LOCAL_RANK", -1))


# 把 learnable_query 定义为一个 Embedding 层
# 词表大小为 1，维度为 (num_of_query, hidden_size) 的 flattened 形式，或者调整形状
# 更简单的方法是自定义一个小 Module
class LearnableQueryLayer(nn.Module):
    def __init__(self, model_num_of_query,hidden_size):
        super().__init__()
        # self.param = nn.Parameter(torch.randn(shape))
        self.param = nn.Embedding(model_num_of_query,hidden_size)
        # self.param2 = nn.Embedding(64,hidden_size)
        self.model_num_of_query = model_num_of_query
        nn.init.normal_(self.param.weight, mean=0.0, std=0.02)
    def forward(self,B):
        query = torch.arange(self.model_num_of_query, device=self.param.weight.device).unsqueeze(0).expand(B,-1)
        query = self.param(query)
        return query


class ConceptSegR1Config(Qwen2_5_VLConfig):
    def __init__(self, num_of_query=None, if_use_qwen_connector=None, if_include_sam=None, **kwargs):
        super().__init__(**kwargs)
        self.num_of_query = num_of_query
        self.if_use_qwen_connector = if_use_qwen_connector




class ConceptSegR1ForConditionalGeneration_qwen2p5(Qwen2_5_VLForConditionalGeneration):
    """
    Conceptseg-R1 model for conditional generation based on Qwen2_5_VL.
    """
    config_class = ConceptSegR1Config

    def __init__(self, config, num_of_query=8,  **kwargs):
        super().__init__(config)
        model_num_of_query = config.num_of_query or num_of_query

        # 在主模型中：
        self.learnable_query = LearnableQueryLayer(model_num_of_query, config.hidden_size)
        self.model_num_of_query = model_num_of_query


        self.connector = Qwen2VLVisionConnectorSimple(depth=4, seq_len=model_num_of_query,
                                                      embed_dim=config.hidden_size)

        # Projection to SAM feature space
        self.proj_to_sam = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 256)  # 1024
        )

        self.post_init()


    def init_sam(self):
        sam = build_sam3_image_model()  #
        self.sam = Sam3Processor(sam)



    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_learnable_query: bool = False,
            sam_text_prompt=None,
            # if_detach_res_loss = True,
            prompt_length = None,
            **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """
        Extended forward method to support learnable query injection.
        """

        images = kwargs.pop("sam_images", None)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if images is not None:
            assert output_hidden_states is True and prompt_length is not None

            implicit_prompt = self.get_sam_embedding(outputs.hidden_states[-1][:, prompt_length:], )

            pred_masks = []
            for i in range(len(implicit_prompt)):
                embedding = implicit_prompt[i]
                if embedding is not None:
                    embedding = embedding.unsqueeze(1)
                inference_state = self.sam.set_image(images[i])
                sam_out = self.sam.set_text_prompt(state=inference_state,
                                                   prompt=[sam_text_prompt[i], embedding])
                mask = sam_out["semantic_seg"]
                pred_masks.append(mask)

            return outputs, pred_masks

        return outputs

    def process_llm_input(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        """
        Convert input_ids to embeddings and append learnable queries at the end.
        """
        if not isinstance(input_ids, torch.LongTensor):
            input_ids = input_ids.to(torch.long)
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        inputs_embeds = torch.cat(
            [inputs_embeds, self.learnable_query(inputs_embeds.size(0))], dim=1
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.size(0), self.model_num_of_query).to(attention_mask)], dim=1
            )
        else:
            attention_mask = torch.ones(inputs_embeds.size(0), inputs_embeds.size(1)).to(inputs_embeds.device)

        return attention_mask, inputs_embeds

    def get_sam_embedding(self, hidden_states):

        hidden_states = hidden_states.detach()
        query  = self.learnable_query(hidden_states.size(0))
        query = self.connector(query,hidden_states)
        implicit_prompt = self.proj_to_sam(query)

        return implicit_prompt

    def postprocess_masks(self, masks, orig_hw):
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks


__all__ = [ "ConceptSegR1ForConditionalGeneration_qwen2p5"]
