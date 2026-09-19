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

"""Gr00tTrainer must surface the shortcut objective's two terms.

The action head returns flow_loss and consistency_loss beside the total, but HF's
Trainer logs only the scalar it optimises. That left the objective's main failure mode
invisible: a consistency term stuck near zero means the model is ignoring dt_level,
and the total loss looks healthy the whole time.

A 5000-step run was completed before this was noticed, and recovering the number
afterwards meant subtracting a control run's loss curve -- which is confounded, since
the two models differ after step 0.
"""

import types

from gr00t.experiment.trainer import Gr00tTrainer
import pytest
import torch


class _Args:
    logging_steps = 10
    local_rank = -1


class _State:
    global_step = 0


def _trainer(outputs: dict) -> tuple[Gr00tTrainer, list[dict]]:
    """A Gr00tTrainer stub exercising compute_loss without building a real Trainer."""
    trainer = object.__new__(Gr00tTrainer)
    trainer.args = _Args()
    trainer.state = _State()
    trainer.action_offset = None
    logged: list[dict] = []

    trainer.log = lambda d, **kw: logged.append(d)
    trainer._nested_gather = lambda t: t  # single process

    loss = torch.tensor(0.5, requires_grad=False)
    # Stand in for Trainer.compute_loss, which this override delegates to.
    trainer_cls_compute = types.MethodType(
        lambda self, model, inputs, return_outputs=False, num_items_in_batch=None: (
            loss,
            outputs,
        ),
        trainer,
    )
    trainer._super_compute_loss = trainer_cls_compute
    return trainer, logged


def _run(trainer, outputs, training=True, step=0):
    """Call the real compute_loss body with the parent call patched out."""
    import transformers.trainer as hf

    model = types.SimpleNamespace(training=training)
    trainer.state.global_step = step

    original = hf.Trainer.compute_loss
    hf.Trainer.compute_loss = lambda self, m, i, return_outputs=False, num_items_in_batch=None: (
        torch.tensor(0.5),
        outputs,
    )
    try:
        return Gr00tTrainer.compute_loss(trainer, model, {}, num_items_in_batch=None)
    finally:
        hf.Trainer.compute_loss = original


class TestShortcutTermsAreLogged:
    def test_both_terms_reach_the_logger(self):
        outputs = {
            "loss": torch.tensor(0.5),
            "flow_loss": torch.tensor(0.4),
            "consistency_loss": torch.tensor(0.1),
        }
        trainer, logged = _trainer(outputs)
        _run(trainer, outputs)

        merged = {k: v for d in logged for k, v in d.items()}
        assert merged["flow_loss"] == pytest.approx(0.4)
        assert merged["consistency_loss"] == pytest.approx(0.1)

    def test_nothing_logged_when_the_objective_is_off(self):
        """A plain flow-matching run returns neither key; the logger must stay quiet
        rather than emit zeros that would look like a collapsed consistency term."""
        outputs = {"loss": torch.tensor(0.5), "action_loss": torch.tensor(0.5)}
        trainer, logged = _trainer(outputs)
        _run(trainer, outputs)

        merged = {k: v for d in logged for k, v in d.items()}
        assert "consistency_loss" not in merged
        assert "flow_loss" not in merged

    def test_quiet_outside_logging_steps(self):
        outputs = {
            "loss": torch.tensor(0.5),
            "flow_loss": torch.tensor(0.4),
            "consistency_loss": torch.tensor(0.1),
        }
        trainer, logged = _trainer(outputs)
        _run(trainer, outputs, step=7)  # logging_steps is 10
        assert not [d for d in logged if "consistency_loss" in d]

    def test_quiet_during_evaluation(self):
        outputs = {
            "loss": torch.tensor(0.5),
            "flow_loss": torch.tensor(0.4),
            "consistency_loss": torch.tensor(0.1),
        }
        trainer, logged = _trainer(outputs)
        _run(trainer, outputs, training=False)
        assert not [d for d in logged if "consistency_loss" in d]

    def test_terms_are_averaged_across_ranks(self):
        """_nested_gather returns one row per rank; both columns must be reduced."""
        outputs = {
            "loss": torch.tensor(0.5),
            "flow_loss": torch.tensor(0.4),
            "consistency_loss": torch.tensor(0.1),
        }
        trainer, logged = _trainer(outputs)
        # Two ranks: this one, plus a peer reporting 0.6 / 0.3.
        trainer._nested_gather = lambda t: torch.cat([t, torch.tensor([[0.6, 0.3]])])
        _run(trainer, outputs)

        merged = {k: v for d in logged for k, v in d.items()}
        assert merged["flow_loss"] == pytest.approx(0.5)
        assert merged["consistency_loss"] == pytest.approx(0.2)
