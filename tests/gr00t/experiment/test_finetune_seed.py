# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The finetune CLI must be able to vary the seed.

Without it every run of a configuration sees the same shard schedule and the same
torch seed, so two runs differ only by CUDA nondeterminism. That is far too small a
perturbation to estimate run-to-run variance from — and the variance has to be known
before a difference between two configurations means anything. On SimplerEnv Bridge
the same configuration run four times spanned 7 to 12 points of success rate per task,
which is the same order as the effects being measured.
"""

import ast
from dataclasses import fields
import inspect
from pathlib import Path

from gr00t.configs.finetune_config import FinetuneConfig


class TestSeedIsExposed:
    def test_finetune_config_has_a_seed(self):
        assert "seed" in {f.name for f in fields(FinetuneConfig)}

    def test_default_preserves_existing_behaviour(self):
        """42 is what the data config and HuggingFace both already used, so adding the
        flag must not move any run that does not pass it."""
        default = {f.name: f.default for f in fields(FinetuneConfig)}["seed"]
        assert default == 42

        from gr00t.configs.data.data_config import DataConfig

        assert {f.name: f.default for f in fields(DataConfig)}["seed"] == 42


class TestSeedReachesTheRun:
    def test_launch_copies_it_onto_the_data_config(self):
        from gr00t.experiment import launch_finetune

        source = Path(launch_finetune.__file__).read_text()
        assert "config.data.seed = ft_config.seed" in source, (
            "seed never leaves FinetuneConfig, so the CLI flag would silently do nothing"
        )

    def test_training_arguments_receive_the_same_value(self):
        """experiment.py already wires TrainingArguments(seed=config.data.seed), which is
        why one field is enough. If that ever changes, the torch seed would stop tracking
        the data seed and a 'replicate' would only vary half of the run."""
        from gr00t.experiment import experiment

        tree = ast.parse(Path(inspect.getsourcefile(experiment)).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "TrainingArguments":
                seeds = [kw for kw in node.keywords if kw.arg == "seed"]
                assert seeds, "TrainingArguments is constructed without a seed"
                assert ast.unparse(seeds[0].value) == "config.data.seed"
                return
        raise AssertionError("no TrainingArguments call found in experiment.py")
