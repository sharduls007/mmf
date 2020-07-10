# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel

from mmf.common.typings import DictConfig
from mmf.models import BaseModel
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert


class BaseTransformer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.config = config
        self.transformer_config = AutoConfig.from_pretrained(
            config.transformer_base, **OmegaConf.to_container(config)
        )

    def build(self):
        """Build the different parts of the multimodal transformer model and
        initializes weights.
        """
        self.build_encoders()
        self.build_embeddings()
        self.build_transformer()
        self.build_heads()
        self.build_losses()

        self.init_weights()

    def get_optimizer_parameters(self, config: DictConfig):
        return get_optimizer_parameters_for_bert(self, config)

    def build_encoders(self):
        """Build any encoders for different input modalities. Encoders are used while
        preprocessing a sample. We the visual_encoder by default for raw image input.

        Example ::

            # For image
            self.vision_encoder = ImageEncoder(self.config)

        """
        pass

    def build_embeddings(self):
        """Build the embeddings for the different input modalities.

        Example ::

            # For text
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=0
            )
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )

            # For image
            self.img_embeddings = nn.Sequential(
                nn.Linear(img_dim, config.hidden_size),
                torch.nn.LayerNorm(config.hidden_size, eps=1e-12),
            )
        """
        pass

    def build_transformer(self):
        """Build the transformer encoder. This uses transformers AutoModel to load a
        pretrained model given the name of the transformer based model. In this base
        implementation we are using only the encoder part of a model. Not all models
        specify the encoder as `self.encoder`, so for some transformer models this
        method needs to be overriden in derived classes to initialize the encoder
        properly.

        Example ::

            self.transformer = AutoModel.from_pretrained(
                "bert-base-uncased",
                config=self.transformer_config,
            ).encoder
        """
        model = AutoModel.from_pretrained(
            self.config.transformer_base,
            config=self.transformer_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )
        self.transformer = model.encoder

    def build_heads(self):
        """Build the different heads for the model. It can be either the pretraining
        head or the classifier heads.

        Example ::

            # For pretraining
            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.transformer_config),
                nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
            )

            # For classification
            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.transformer_config),
                nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
            )
        """
        pass

    def build_losses(self):
        """Initialize the losses for pretraining. For example MLM, MIM etc.

        Example ::

            self.mlm_loss = CrossEntropyLoss(ignore_index=-1)
        """
        pass

    def _init_weights(self, module):
        """Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.module.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings
        if required.
        """
        pass

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.transformer_base is None:
                # No pretrained model, init weights
                self.apply(self._init_weights)

        # Tie weights if required
        self.tie_weights()

    def preprocess_sample(self, sample_list: Dict[str, Any]):
        """Preprocess the sample_list. Generate image input ids, image position ids,
        image mask. Create a attention mask for all the masks for different modalities.
        """
        # TODO(vedanuj): Make this method simpler
        # Image feature(img_input_ids) and position(img_pos_ids) tokens
        if "image_feature_0" in sample_list:
            sample_list.img_input_ids = sample_list.image_feature_0
            sample_list.img_pos_ids = sample_list.image_info_0.bbox
        else:
            sample_list.img_input_ids = self.vision_encoder(sample_list.image)
            sample_list.img_pos_ids = sample_list.img_input_ids.new_tensor(
                torch.arange(0, sample_list.img_input_ids.size(1), dtype=torch.long)
                .unsqueeze(0)
                .expand(
                    sample_list.img_input_ids.size(0), sample_list.img_input_ids.size(1)
                )
                .unsqueeze(-1)
            )

        # Image mask
        sample_list.image_mask = torch.ones_like(
            sample_list.img_input_ids[:, :, 0], dtype=torch.long
        )

        # Attention mask
        sample_list.attention_mask = torch.cat(
            (sample_list.input_mask, sample_list.image_mask), dim=-1
        )
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        if sample_list.lm_label_ids is not None:
            assert sample_list.masked_lm_labels.size(-1) == sample_list.input_mask.size(
                -1
            )
            new_lm_labels = torch.ones_like(sample_list.attention_mask) * -1
            size_masked_lm_labels = sample_list.masked_lm_labels.size()
            assert len(size_masked_lm_labels) == 2
            new_lm_labels[
                : size_masked_lm_labels[0], : size_masked_lm_labels[1]
            ] = sample_list.masked_lm_labels
            sample_list.masked_lm_labels = new_lm_labels

    def forward(self, sample_list: Dict[str, Any]):
        pass

    def postprocess_output(self, output: List[Tensor]) -> Dict[str, Tensor]:
        """Postprocessing the output from the transformer head, for pretraining
        it's the output of the pretrain head and for classification its the output
        of the classsification head. Calculate lossses on pretraining output or
        model output scores.

        Returns:
            Dict[str, Tensor]: Dict containing scores or losses
        """
        return output
