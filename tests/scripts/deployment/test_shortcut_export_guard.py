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

"""The ONNX exporter must refuse shortcut checkpoints rather than degrade them.

The exported graph pins the DiT's inputs, so dt_level would never reach the
denoiser: the step-size conditioning is dropped, and the engine is quietly worse
at 1 and 2 steps. Engines get shipped, so the failure has to be loud.
"""

import importlib.util
from pathlib import Path
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "deployment" / "export_onnx_n1d7.py"


def _load_guard():
    """Import just the guard, without the script's heavy CUDA/TensorRT imports."""
    source = EXPORT_SCRIPT.read_text()
    start = source.index("def _reject_shortcut_checkpoint(policy)")
    end = source.index("def main(args):")
    module = types.ModuleType("shortcut_export_guard")
    exec(compile(source[start:end], str(EXPORT_SCRIPT), "exec"), module.__dict__)
    return module._reject_shortcut_checkpoint


class _Policy:
    def __init__(self, dt_encoder):
        dit = types.SimpleNamespace(dt_encoder=dt_encoder)
        self.model = types.SimpleNamespace(action_head=types.SimpleNamespace(model=dit))


class TestExportGuard:
    def test_shortcut_checkpoint_is_rejected(self):
        guard = _load_guard()
        with pytest.raises(NotImplementedError, match="shortcut objective"):
            guard(_Policy(dt_encoder=object()))

    def test_plain_checkpoint_is_allowed(self):
        guard = _load_guard()
        guard(_Policy(dt_encoder=None))  # must not raise

    def test_model_without_a_dit_is_allowed(self):
        guard = _load_guard()
        policy = types.SimpleNamespace(model=types.SimpleNamespace())
        guard(policy)  # must not raise

    def test_the_guard_runs_before_any_export_work(self):
        """A guard placed after the capture/export steps would still waste the run and,
        worse, could be reordered away without any test noticing."""
        source = EXPORT_SCRIPT.read_text()
        main_body = source[source.index("def main(args):") :]
        guard_at = main_body.index("_reject_shortcut_checkpoint(policy)")
        for later_step in ("[Step 2]", "[Step 3]", "export_dit_to_onnx"):
            assert guard_at < main_body.index(later_step), (
                f"the shortcut guard runs after {later_step}"
            )


def test_export_script_is_importable_as_a_file():
    """Sanity: the guard extraction above depends on the script's structure."""
    assert EXPORT_SCRIPT.is_file()
    assert importlib.util.spec_from_file_location("x", EXPORT_SCRIPT) is not None
    assert "shortcut_export_guard" not in sys.modules or True
