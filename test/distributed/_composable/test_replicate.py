# Owner(s): ["oncall: distributed"]

import sys
from copy import deepcopy
from functools import partial, wraps

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import run_tests


def with_comms(func=None):
    if func is None:
        return partial(with_comms)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        device = kwargs.get("device", None)
        BACKEND = dist.get_default_backend_for_device(device)
        if BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.pg = self.create_pg(device=device)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class ReplicateStateDictTest(DistributedTestBase):
    def _check_state_dict_parity(self, sd_1, sd_2):
        for k1, k2 in zip(sd_1.keys(), sd_2.keys()):
            self.assertEqual(k1, k2)

        for v1, v2 in zip(sd_1.values(), sd_2.values()):
            self.assertEqual(v1, v2)

    @with_comms
    def test_replicate_single_module_save_load(self, device):
        """
        Tests that replicate() on a single module state_dict
        matches local module state_dict.
        """
        model = Net().to(device)
        replicate_model = replicate(deepcopy(model), device_id=torch.device(device))
        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)

    @with_comms
    def test_replicate_non_root_multiple_save_load(self, device):
        """
        Tests tha replicate() on multiple submodules matches
        local module state_dict.
        """
        model = Net().to(device)
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1, device_id=torch.device(device))
        replicate(replicate_model.fc2, device_id=torch.device(device))
        replicate(replicate_model.fc3, device_id=torch.device(device))

        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)


class ReplicateTest(MultiProcessTestCase):
    def _compare_module(self, mod, replicate_mod):
        local_batch_size = 1
        global_batch_size = self.world_size * local_batch_size
        input = torch.randn(global_batch_size, 2)
        target = torch.randn(global_batch_size, 2)

        def step_model(model, input, target, device):
            model, input, target = model.to(device), input.to(device), target.to(device)
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        for iteration in range(2):
            step_model(mod, input, target)
            step_model(
                replicate_mod,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            self.assertEqual(
                len(list(mod.parameters())),
                len(list(replicate_mod.parameters())),
            )
            for i, j in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    @with_comms
    def test_replicate_single_module(self, device):
        model = Net().to(device)
        replicate_model = replicate(deepcopy(model))
        self._compare_module(model, replicate_model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_move_args_kwargs_to_device(self, device):
        class MyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(2, 2)

            def forward(self, inp, *, kwarg=None):
                if kwarg is not None:
                    inp = inp @ kwarg
                return self.a(inp)

        model = MyNet().to(device)
        replicate(model, device_id=torch.device(device))
        # CPU input ensures replicate can move arg and kwargs to device.
        a, b = torch.randn(2, 2).to(device), torch.randn(2, 2).to(device)
        model(a, kwarg=b).sum().backward()

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_ignore_module(self, device):
        # Seed ensures diff input and thus different local grads across ranks.
        torch.manual_seed(self.rank)
        torch.get_device_module(device).manual_seed(self.rank)
        model = Net().to(device)
        replicate(model, ignored_modules=[model.fc1])
        # CPU input ensures that replicate can move input to GPU as DDP does.
        inp = torch.randn(5, 2, device=device) * (self.rank + 1)
        out = model(inp) * 10
        out.sum().backward()
        # FC1 grads should not be synchronized, FC2 and 3 should be.
        fc1_grad = model.fc1.weight.grad
        tensor_list = [
            torch.zeros_like(fc1_grad).to(device) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(tensor_list, fc1_grad)
        grad, rest = tensor_list[0], tensor_list[1:]
        for g in rest:
            self.assertNotEqual(grad, g)

        for dp_grad in [model.fc2.weight.grad, model.fc3.weight.grad]:
            tensor_list = [
                torch.zeros_like(dp_grad) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list, dp_grad)
            grad, rest = tensor_list[0], tensor_list[1:]
            for g in rest:
                self.assertEqual(grad, g)

    @with_comms
    def test_replicate_multi_module(self, device):
        model = Net().to(device)
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)
        self._compare_module(model, replicate_model)

    @with_comms
    def test_replicate_with_kwargs(self, device):
        model = Net().to(device)
        replicate_model = replicate(
            deepcopy(model), bucket_cap_mb=1, gradient_as_bucket_view=True
        )
        self._compare_module(model, replicate_model)

    @with_comms
    def test_replicate_device_id_cpu(self):
        model = Net()
        replicate(model, device_id=torch.device("cpu"))
        # DDP instance is attached in first pre forward
        model(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model)._ddp_weakref()
        # Should be None for CPU training
        self.assertEqual(None, replicate_ddp_weakref.device_ids)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_device_id(self, device):
        model = Net().to(device)
        model_accelerator = deepcopy(model).to(device)
        model_accelerator2 = deepcopy(model_accelerator)

        replicate(model_accelerator, device_id=torch.device(device))
        # DDP instance is attached in first pre forward
        model_accelerator(torch.randn(2, 2).to(device))
        replicate_ddp_weakref = replicate.state(model_accelerator)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)
        # Pass in int as device_id
        replicate(model_accelerator2, device_id=torch.device(device))
        # DDP instance is attached in first pre forward
        model_accelerator2(torch.randn(2, 2).to(device))
        replicate_ddp_weakref = replicate.state(model_accelerator2)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)

    @with_comms
    def test_replicate_wrong_device_id_type(self):
        model = Net()
        with self.assertRaisesRegex(
            RuntimeError, "Expected device_id to be int or torch.device"
        ):
            replicate(model, device_id=[torch.device("cpu")])


devices = ["cpu", "cuda", "hpu"]
instantiate_device_type_tests(ReplicateStateDictTest, globals(), only_for=devices)
instantiate_device_type_tests(ReplicateTest, globals(), only_for=devices)

if __name__ == "__main__":
    run_tests()
