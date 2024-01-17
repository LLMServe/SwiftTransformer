"""
# converter_lib.py - Libraries & functions for converting weights from OPT/LLaMA2 to SwiftTransformer's format

## Workflow

For generalization, the workflow of converting weights is divided into 3 steps:

### 1. Reading & resharding.

This step is model-specific.

The weights are read from the input file and resharded if necessary (i.e. when
the weights are stored in multiple shards) into a dict called `tensor_dict`.

While the actual resharding process is model-specific, the whole process in this
step can be modeled as follows:

- convert_lib.py (i.e. this file) provides a function, `reshardWeight`, which takes
  three arguments: files, preprocess_script, and tensor_merge_func.
    - files is a list of file paths. Each file contains a dict of tensors.
    - preprocess_script is a function that takes a dict of tensors and returns
        a dict of tensors. This function does simple jobs like, when converting
        from OPT, it is just `lambda x: x["model"]`
    - tensor_merge_func is a function that takes a key, a list of sharded tensors,
        and returns a merged tensor. This function does the actual resharding, which
        is model-specific.  
- The `reshardWeight` function first opens all files and classifies all tensors based
    on their keys (in sharded weights, the key is the same for all shards for one
    particular weight). Then it calls `tensor_merge_func` for each key and the list
    of tensors with that key. The return value of `tensor_merge_func` is then stored
    in `tensor_dict` with the same key. (Somehow this is similar to the MapReduce)

### 2. Preprocessing.

This step is highly model-specific.

The weights are preprocessed so that the following weight conversion step can
be done in a generic way. After this step, the variable `tensor_dict` should be
a dict of tensors, and the shape and meaning of tensors inside it should be the same as OPT's.

### 3. DivideWeightAndSave.

This step is model-agnotic.

This step can be further divided into 2 substeps:

3.1. Renaming. After step 2, tensors have meanings similar to OPT's, but the names
    may vary. This step takes an argument, nameTranslator, which is a function that
    takes a name and returns a new name (or None if this weight should be ignored).
    After this step there should be the following tensors in `tensor_dict`:
    - decoder.embed_tokens.weight
    - decoder.embed_positions.weight
    - decoder.layer_norm.weight
    - decoder.layer_norm.bias
    - decoder.layers.{layer_id}.fc1.weight
    ...

    For more details (include names and dimensions) about the tensors, please refer
    to regexs and comments in `divideWeightAndSave`

3.2 Divide and save. For tensor parallelism, some tensors may need to be divided
    along one particular dimension and saved to multiple files, while other tensors
    should be saved directly. This step accomplishes this.

Here is an illustration of the workflow:

   FILES     preprocess  tensor_merge
             _script     _func

shard.00.pt --------------->+                 preprocessing             renaming
                            |
shard.01.pt --------------->+---> tensor_dict ------------> tensor_dict -------> standarized_tensor_dict
                            |                                                      |
shard.02.pt --------------->+                                                      | divide and save
                            |                                                      V
shard.03.pt --------------->+                                                    output files
"""

import os, re, argparse
from collections.abc import Callable
from typing import Optional, List

import tqdm
from torch import nn
import torch

# reshardWeight - Load weight from multiple shards and merge them.
def reshardWeight(
        files: List[str],
        preprocess_script: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        tensor_merge_func: Callable[[str, List[torch.Tensor]], Optional[torch.Tensor]]
    ) -> dict[str, torch.Tensor]:

    files = sorted(files, key=lambda x: list(map(int, re.findall(r"\d+", x))))
    print(f"Found {len(files)} model parallel parts ({files[0]} to {files[-1]})")

    sharded_tensors: dict[str, List[torch.Tensor]] = {}
    print(f"Loading files")
    for i, file in tqdm.tqdm(enumerate(files)):
        print(f"Loading {file}")
        state_dict = torch.load(file, torch.device("cpu"))
        state_dict = preprocess_script(state_dict)
        for key, tensor in state_dict.items():
            if key not in sharded_tensors:
                sharded_tensors[key] = []
            sharded_tensors[key].append(tensor)
        del state_dict  # Free memory
    
    print(f"Merging tensors")
    result = {}
    keys = list(sharded_tensors.keys())
    for key in tqdm.tqdm(keys):
        tensors = sharded_tensors[key]
        merged_tensor = tensor_merge_func(key, tensors)
        if merged_tensor != None:
            result[key] = merged_tensor
        del sharded_tensors[key]    # Free memory
    return result

# Save a tensor to a file
# The file can be loaded by torch.jit.load(). The file contains a single
# tensor whose key is "key"
def saveTensorToFile(filename: str, key: str, tensor: torch.Tensor):
    # https://discuss.pytorch.org/t/load-tensors-saved-in-python-from-c-and-vice-versa/39435/8
    class TensorContainer(nn.Module):
        def __init__(self, key: str, tensor: torch.Tensor):
            super().__init__()
            setattr(self, key, tensor)

    container = TensorContainer(key, tensor.clone()) # clone() is needed or the whole (undivided) tensor will be saved
    torch.jit.script(container).save(filename)

# The following functions are name translators
# They are used to convert the names of the weights from different models (e.g. LLaMA2) to the format used by SwiftTransformer
# The name translator is a function that takes a name and returns a new name (or None if this weight should be ignored)
# The new name is the name of the weight in SwiftTransformer (following OPT's naming convention)
def optNameTranslator(name: str) -> Optional[str]:
    if name == "decoder.version":
        return None
    return name

def llama2NameTranslator(name: str) -> Optional[str]:
    ignore_regex = re.compile(r"rope.freqs")
    if ignore_regex.match(name):
        return None
    name_mapping_table = [
        (re.compile(r"layers.(?P<layer>\d+).attention.wo.weight"), "decoder.layers.{layer}.self_attn.out_proj.weight"),
        (re.compile(r"layers.(?P<layer>\d+).feed_forward.w(?P<fc>\d+).weight"), "decoder.layers.{layer}.fc{fc}.weight"),
        (re.compile(r"layers.(?P<layer>\d+).attention_norm.weight"), "decoder.layers.{layer}.self_attn_layer_norm.weight"),
        (re.compile(r"layers.(?P<layer>\d+).ffn_norm.weight"), "decoder.layers.{layer}.final_layer_norm.weight"),
        (re.compile(r"layers.(?P<layer>\d+).attention.wqkv.weight"), "decoder.layers.{layer}.self_attn.qkv_proj.weight"),
        (re.compile(r"tok_embeddings.weight"), "decoder.embed_tokens.weight"),
        (re.compile(r"output.weight"), "decoder.output_projection.weight"),
        (re.compile(r"norm.weight"), "decoder.layer_norm.weight")
    ]
    for (regex, newname) in name_mapping_table:
        match = regex.match(name)
        if match:
            return newname.format(**match.groupdict())
    assert False, f"Cannot find a match for {name} when translating name"

# divideWeightAndSave: The last step in convertWeight(). It takes a tensor_dict and a name translator,
# translate the names of the weights, divide the weights and save them to files
def divideWeightAndSave(output_dir: str, tensor_dict: dict[str, torch.Tensor], name_translator: Callable[[str], Optional[str]], num_q_heads: int, head_dim: int):
    # divideTensorAndStore: divide a tensor along a dimension into 8 pieces and save the divided tensors to files
    def divideTensorAndStore(new_key, value, dim: int):
        assert value.size(dim) % 8 == 0, f"Cannot divide {new_key} along dim={dim} because the size of the dimension is not divisible by 8"
        value = torch.split(value, value.size(dim) // 8, dim=dim)
        # save the tensor to file
        for i in range(len(value)):
            filename = f"{output_dir}/{new_key}.tp_{i}.pt"
            saveTensorToFile(filename, new_key, value[i])

    # storeQKVKernelOrBias: divide the QKV kernel or bias and save them to files
    def storeQKVKernelOrBias(key: str, qkvs: torch.Tensor, split_dim: int):
        qkvs = qkvs.view(qkvs.size(0), -1, head_dim) if split_dim == 1 else qkvs.view(-1, head_dim)
        num_kv_heads = (qkvs.size(split_dim)-num_q_heads) // 2
        # qkvs: [hidden_size, (num_q_heads+2*num_kv_heads), head_dim] (when converting QKV kernel)
        # or, [(num_q_heads+2*num_kv_heads), head_dim] (when converting QKV bias)

        # Deal with cases where num_q_heads or num_kv_heads is not divisible by 8, like in OPT-125M
        q_heads_in_each_tensor = num_q_heads // 8 if num_q_heads%8 == 0 else [ num_q_heads // 8 ] * 7 + [ num_q_heads - (num_q_heads // 8) * 7 ]
        kv_heads_in_each_tensor = num_kv_heads // 8 if num_kv_heads%8 == 0 else [ num_kv_heads // 8 ] * 7 + [ num_kv_heads - (num_kv_heads // 8) * 7 ]

        # split qkvs into 3 parts: q, k, v
        qs = qkvs.narrow(split_dim, 0, num_q_heads).split(q_heads_in_each_tensor, split_dim)
        ks = qkvs.narrow(split_dim, num_q_heads, num_kv_heads).split(kv_heads_in_each_tensor, split_dim)
        vs = qkvs.narrow(split_dim, num_q_heads+num_kv_heads, num_kv_heads).split(kv_heads_in_each_tensor, split_dim)

        # save the tensors to files
        for i in range(8):
            filename = f"{output_dir}/{key}.tp_{i}.pt"
            saveTensorToFile(filename, key, torch.cat([qs[i], ks[i], vs[i]], dim=split_dim))

    # The following regexs define tensors in the standarized tensor dict
    # Note that not all tensors listed below need to present. For example,
    # LLaMA2 does not have decoder.embed_positions.weight since it used RoPE.

    # Tensors that need to be divided along dim=0
    to_divide_by_dim0_regex = re.compile("|".join([
        "decoder.layers.(\d+).fc1.weight",      # [ffn_inter_dim, hidden_size]
        "decoder.layers.(\d+).fc1.bias",        # [ffn_inter_dim]
        "decoder.layers.(\d+).fc3.weight",      # [ffn_inter_dim, hidden_size]
        "decoder.layers.(\d+).self_attn.out_proj.weight",   # [(num_q_heads*head_dim), hidden_size]
        "decoder.layers.(\d+).self_attn.qkv_proj.bias"      # [(num_q_heads+2*num_kv_heads)*head_dim]
                                         ]))
    # Tensors that need to be divided along dim=1
    to_divide_by_dim1_regex = re.compile("|".join([
        "decoder.layers.(\d+).fc2.weight"       # [hidden_size, ffn_inter_dim]
                                         ]))
    # Tensors that need to be replicated among all tensor parallel workers
    to_replicate_regex = re.compile("|".join([
        "decoder.embed_tokens.weight",          # [vocab_size, hidden_size]
        "decoder.embed_positions.weight",       # [max_positions, hidden_size]
        "decoder.output_projection.weight",     # [vocab_size, hidden_size]
        "decoder.layer_norm.(weight|bias)",     # [hidden_size] (The final layernorm)
        "decoder.layers.(\d+).self_attn_layer_norm.(weight|bias)",  # [hidden_size]
        "decoder.layers.(\d+).final_layer_norm.(weight|bias)",      # [hidden_size]
        "decoder.layers.(\d+).self_attn.out_proj.bias",             # [hidden_size]
        "decoder.layers.(\d+).fc2.bias"                    # [hidden_size]
                                    ]))
    
    # And we have two special tensors:
    #   - decoder.layers.{layer_id}.self_attn.qkv_proj.weight   [hidden_size, (num_q_heads+2*num_kv_heads)*head_dim]
    #   - decoder.layers.{layer_id}.self_attn.qkv_proj.bias     [(num_q_heads+2*num_kv_heads)*head_dim]
    # which need to be handled separately

    for key, tensor in tqdm.tqdm(tensor_dict.items()):
        new_key = name_translator(key)
        if new_key == None:
            continue
        if "self_attn.qkv_proj.weight" in new_key:
            storeQKVKernelOrBias(new_key, tensor, split_dim=1)
        elif "self_attn.qkv_proj.bias" in new_key:
            storeQKVKernelOrBias(new_key, tensor, split_dim=0)
        elif to_divide_by_dim0_regex.search(new_key):
            divideTensorAndStore(new_key, tensor, dim=0)
        elif to_divide_by_dim1_regex.search(new_key):
            divideTensorAndStore(new_key, tensor, dim=1)
        elif to_replicate_regex.search(new_key):
            filename = f"{output_dir}/{new_key}.pt"
            saveTensorToFile(filename, new_key, tensor)
        else:
            assert False, f"Cannot find a match for {new_key} when dispatching tensors"

# convertWeight - The main function that converts the weights
def convertWeight(output_dir: str, tensor_dict: dict[str, torch.Tensor], dtype: torch.dtype, model_name: str):
    # Change dtype
    for key in tensor_dict:
        tensor_dict[key] = tensor_dict[key].to(dtype)

    # Preprocessing
    num_q_heads = 0
    head_dim = 0
    if model_name == "opt":
        regex = re.compile(r"decoder.layers.(\d+).fc1.weight")
        num_layers = max(int(regex.findall(x)[0]) for x in filter(regex.match, tensor_dict)) + 1
        
        def _kvq_to_qkv(t: torch.Tensor) -> torch.Tensor:
            t = t.view(3, t.size(0) // 3, *t.size()[1:])
            t = torch.cat([t[2:], t[:2]], dim=0)
            return t if t.ndim == 2 else t.permute(2, 0, 1).contiguous()
        
        ffn_inter_dim = tensor_dict["decoder.layers.0.fc1.bias"].size(0)
        head_dim = \
            64 if ffn_inter_dim <= 8192 else \
            80 if ffn_inter_dim == 10240 else \
            128
        num_q_heads = tensor_dict["decoder.layer_norm.weight"].size(0) // head_dim

        tensor_dict["decoder.embed_positions.weight"] = tensor_dict["decoder.embed_positions.weight"][2:]
        tensor_dict["decoder.output_projection.weight"] = tensor_dict["decoder.embed_tokens.weight"]

        # Transpose out_proj.weight and qkv_proj.weight
        for i in range(num_layers):
            tensor_dict[f"decoder.layers.{i}.self_attn.out_proj.weight"] = \
                tensor_dict[f"decoder.layers.{i}.self_attn.out_proj.weight"].T.contiguous()
            tensor_dict[f"decoder.layers.{i}.self_attn.qkv_proj.weight"] = \
                _kvq_to_qkv(tensor_dict[f"decoder.layers.{i}.self_attn.qkv_proj.weight"])
            tensor_dict[f"decoder.layers.{i}.self_attn.qkv_proj.bias"] = \
                _kvq_to_qkv(tensor_dict[f"decoder.layers.{i}.self_attn.qkv_proj.bias"])
        
    elif model_name == "llama2":
        regex = re.compile(r"layers.(\d+).attention.wq.weight")
        num_layers = max(int(regex.findall(x)[0]) for x in filter(regex.match, tensor_dict)) + 1

        head_dim = 128
        num_q_heads = tensor_dict["layers.0.attention.wq.weight"].size(0) // head_dim

        # Coallesce wq, qk, qv into one tensor, layers.{i}.attention.wqkv.weight
        for i in range(num_layers):
            q = tensor_dict[f"layers.{i}.attention.wq.weight"].T  # [hidden_size, num_q_heads*head_dim]
            k = tensor_dict[f"layers.{i}.attention.wk.weight"].T  # [hidden_size, num_kv_heads*head_dim]
            v = tensor_dict[f"layers.{i}.attention.wv.weight"].T  # [hidden_size, num_kv_heads*head_dim]
            wqkv = torch.cat([q, k, v], dim=1)    # [hidden_size, (num_q_heads+2*num_kv_heads)*head_dim]
            tensor_dict[f"layers.{i}.attention.wqkv.weight"] = wqkv
            del tensor_dict[f"layers.{i}.attention.wq.weight"]
            del tensor_dict[f"layers.{i}.attention.wk.weight"]
            del tensor_dict[f"layers.{i}.attention.wv.weight"]

        # Transpose wo
        for i in range(num_layers):
            wo = tensor_dict[f"layers.{i}.attention.wo.weight"].T.contiguous()  # [num_q_heads*head_dim, hidden_size]
            tensor_dict[f"layers.{i}.attention.wo.weight"] = wo
    

    # The final step: divide the weights and save them to files
    assert num_q_heads > 0, "num_q_heads must be greater than 0"
    divideWeightAndSave(output_dir, tensor_dict, {
        "opt": optNameTranslator,
        "llama2": llama2NameTranslator
    }[model_name], num_q_heads, head_dim)
