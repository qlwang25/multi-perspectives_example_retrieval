# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING
from transformers.utils.file_utils import _LazyModule


_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
    "tokenization_gpt2": ["GPT2Tokenizer"],
}

_import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]
_import_structure["modeling_gpt2"] = [
    "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
    "GPT2LMHeadModel",
    "GPT2ForTokenClassification",
    "GPT2ForSequenceClassification",
    "GPT2Model",
    "GPT2PreTrainedModel",
]


if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig
    from .tokenization_gpt2 import GPT2Tokenizer

    from .tokenization_gpt2_fast import GPT2TokenizerFast

    from .modeling_gpt2 import (
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        GPT2LMHeadModel,
        GPT2ForTokenClassification,
        GPT2ForSequenceClassification,
        GPT2Model,
        GPT2PreTrainedModel,
    )

else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
