# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"MoE (Mixture of Experts) integration file"

from ..activations import ACT2FN
from ..utils import is_accelerate_available, is_aqlm_available, is_torch_available

if is_torch_available():
    import torch.nn as nn

from moetools.moe_mlp import MoeMLP
from moetools.moe_linear import ClusteredLinear
from moetools.clusteringutils import torchPCA, torchKMeans


# MOE_FUSED_MAPPINGS = {
#     "llama": modeling_llama
# }

# MOE_FUSED_MAPPINGS = {
#     "llama": getattr(__import__("llama", fromlist=["LlamaMLP"]), "LlamaMLP")
# }

# class MoeMLP(nn.Module):

#     def __init__(
#             self, 
#             hidden_size, 
#             intermediate_size,
#             model_config, 
#             num_local_experts, 
#             num_experts_per_tok
#     ):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size

#         self.num_experts = num_local_experts
#         self.top_k = num_experts_per_tok

#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
#         self.act_fn = ACT2FN[model_config.hidden_act]

#     def forward(self, x):
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#         return down_proj



def replace_with_moe_mlp(
    model,
    model_config=None,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
    mlp_blocks_not_to_replace=None
):
    """
    Public method that recursively replaces the Linear layers of the given model with AQLM quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AqlmConfig`):
            The quantization config object that contains the quantization parameters.
        linear_weights_not_to_quantize (`list[str]`, *optional*):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    if not is_accelerate_available():
        raise ValueError("MoeMLPBlock requires Accelerate to be installed: `pip install accelerate`")

    if mlp_blocks_not_to_replace is None:
        mlp_blocks_not_to_replace = []

    from accelerate import init_empty_weights

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if current_key_name[-1] == "mlp":
            with init_empty_weights():

                model._modules[name] = MoeMLP(
                    hidden_size=module.hidden_size, 
                    intermediate_size=module.intermediate_size,
                    model_config=model_config,
                    num_local_experts=quantization_config.num_local_experts,
                    num_experts_per_tok=quantization_config.num_experts_per_tok
                )

                # model._modules[name] = QuantizedLinear(
                #     in_features,
                #     out_features,
                #     bias=module.bias is not None,
                #     in_group_size=quantization_config.in_group_size,
                #     out_group_size=quantization_config.out_group_size,
                #     num_codebooks=quantization_config.num_codebooks,
                #     nbits_per_codebook=quantization_config.nbits_per_codebook,
                # )
                has_been_replaced = True

                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_moe_mlp(
                module,
                model_config=model_config,
                quantization_config=quantization_config,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
                mlp_blocks_not_to_replace=mlp_blocks_not_to_replace
            )
            # _, has_been_replaced = replace_with_moe_mlp(
            #     module,
            #     quantization_config=quantization_config,
            #     linear_weights_not_to_quantize=linear_weights_not_to_quantize,
            #     current_key_name=current_key_name,
            #     has_been_replaced=has_been_replaced,
            # )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_clustered_linear(
    model,
    linear_layers_config=None,
    current_key_name=None,
    has_been_replaced=False
):
    """
    Public method that recursively replaces the Linear layers of the given model with AQLM quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AqlmConfig`):
            The quantization config object that contains the quantization parameters.
        linear_weights_not_to_quantize (`list[str]`, *optional*):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    if not is_accelerate_available():
        raise ValueError("MoeMLPBlock requires Accelerate to be installed: `pip install accelerate`")

    # if mlp_blocks_not_to_replace is None:
    #     mlp_blocks_not_to_replace = []
    import torch
    from accelerate import init_empty_weights
    

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            if "mlp" in current_key_name:
                in_features = module.in_features
                out_features = module.out_features
                num_clusters = linear_layers_config['num_clusters']
                clustered_layer = ClusteredLinear(
                    in_features, out_features, 
                    num_clusters=num_clusters
                )

                clustered_layer.pruning_masks = torch.empty(
                    (num_clusters, out_features, in_features),
                    dtype=torch.bool
                )
                clustered_layer.mode = 'test'

                with init_empty_weights():
                    # for i in range(0, num_clusters):
                    #     if i == 0:
                    #         linear_layer = None
                    #     else:
                    #         linear_layer = nn.Linear(in_features, out_features, bias=False)
                    #     clustered_layer.add_layer(linear_layer)
                    
                    hidden_dim = in_features
                    reduction_factor = linear_layers_config['pca_reduction_factor']
                    n_components = hidden_dim // reduction_factor


                    clustered_layer.pca_model = torchPCA(
                        hidden_dim=hidden_dim,
                        n_components=n_components
                    )

                    clustered_layer.kmeans_model = torchKMeans(
                        hidden_dim=n_components,
                        n_clusters=num_clusters
                    )


                    model._modules[name] = clustered_layer

                    # model._modules[name] = QuantizedLinear(
                    #     in_features,
                    #     out_features,
                    #     bias=module.bias is not None,
                    #     in_group_size=quantization_config.in_group_size,
                    #     out_group_size=quantization_config.out_group_size,
                    #     num_codebooks=quantization_config.num_codebooks,
                    #     nbits_per_codebook=quantization_config.nbits_per_codebook,
                    # )
                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_clustered_linear(
                module,
                linear_layers_config=linear_layers_config,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced
            )
            # _, has_been_replaced = replace_with_moe_mlp(
            #     module,
            #     quantization_config=quantization_config,
            #     linear_weights_not_to_quantize=linear_weights_not_to_quantize,
            #     current_key_name=current_key_name,
            #     has_been_replaced=has_been_replaced,
            # )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

