import argparse
import os
import sys
import time
import warnings
from functools import partial

import experiment as exp
import lab as B
# import lab.tensorflow
import neuralprocesses.torch as nps
import numpy as np
import torch
import wbml.out as out
from matrix.util import ToDenseWarning
from wbml.experiment import WorkingDirectory

__all__ = ["main"]

warnings.filterwarnings("ignore", category=ToDenseWarning)



def get_model(**kw_args):
    """Builds the model and restores the weights"""

    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="*", default=["_experiments"])
    parser.add_argument("--subdir", type=str, nargs="*")
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument("--dim-y", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model",
        choices=[
            "cnp",
            "gnp",
            "np",
            "acnp",
            "agnp",
            "anp",
            "convcnp",
            "convgnp",
            "convnp",
            "fullconvgnp",
            # Experiment-specific architectures:
            "convcnp-mlp",
            "convgnp-mlp",
            "convcnp-multires",
            "convgnp-multires",
        ],
        default="convcnp",
    )
    parser.add_argument(
        "--arch",
        choices=[
            "unet",
            "unet-sep",
            "unet-res",
            "unet-res-sep",
            "conv",
            "conv-sep",
            "conv-res",
            "conv-res-sep",
        ],
        default="unet",
    )
    parser.add_argument(
        "--data",
        choices=exp.data,
        default="eq",
    )
    parser.add_argument("--objective", choices=["loglik", "elbo"], default="loglik")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--resume-at-epoch", type=int)
    parser.add_argument("--train-fast", action="store_true")
    parser.add_argument("--check-completed", action="store_true")
    parser.add_argument("--unnormalised", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-last", action="store_true")
    parser.add_argument("--evaluate-fast", action="store_true")
    parser.add_argument("--evaluate-num-plots", type=int, default=1)
    parser.add_argument(
        "--evaluate-objective",
        choices=["loglik", "elbo"],
        default="loglik",
    )
    parser.add_argument("--evaluate-num-samples", type=int, default=512)
    parser.add_argument("--evaluate-batch-size", type=int, default=8)
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--ar", action="store_true")
    parser.add_argument("--also-ar", action="store_true")
    parser.add_argument("--no-ar", action="store_true")
    parser.add_argument("--experiment-setting", type=str, nargs="*")

    if kw_args:
        # Load the arguments from the keyword arguments passed to the function.
        # Carefully convert these to command line arguments.
        args = parser.parse_args(
            sum(
                [
                    ["--" + k.replace("_", "-")] + ([str(v)] if v is not True else [])
                    for k, v in kw_args.items()
                ],
                [],
            )
        )
    else:
        args = parser.parse_args()

    # Remove the architecture argument if a model doesn't use it.
    if args.model not in {
        "convcnp",
        "convgnp",
        "convnp",
        "fullconvgnp",
    }:
        del args.arch

    # Remove the dimensionality specification if the experiment doesn't need it.
    if not exp.data[args.data]["requires_dim_x"]:
        del args.dim_x
    if not exp.data[args.data]["requires_dim_y"]:
        del args.dim_y

    # Ensure that `args.experiment_setting` is always a list.
    if not args.experiment_setting:
        args.experiment_setting = []

    # Determine settings for the setup of the script.
    suffix = ""
    observe = False
    if args.check_completed or args.no_action or args.load:
        observe = True
    elif args.evaluate:
        suffix = "_evaluate"
        if args.ar:
            suffix += "_ar"
    else:
        # The default is training.
        suffix = "_train"

    # Setup script.
    if not observe:
        out.report_time = True
    wd = WorkingDirectory(
        *args.root,
        *(args.subdir or ()),
        args.data,
        *((f"x{args.dim_x}_y{args.dim_y}",) if hasattr(args, "dim_x") else ()),
        args.model,
        *((args.arch,) if hasattr(args, "arch") else ()),
        args.objective,
        log=f"log{suffix}.txt",
        diff=f"diff{suffix}.txt",
        observe=observe,
    )

    # Determine which device to use. Try to use a GPU if one is available.
    if args.device:
        device = args.device
    elif args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    B.set_global_device(device)
    # Maintain an explicit random state through the execution.
    state = B.create_random_state(torch.float32, seed=0)

    # General config.
    config = {
        "default": {
            "epochs": None,
            "rate": None,
            "also_ar": False,
        },
        "epsilon": 1e-8,
        "epsilon_start": 1e-2,
        "cholesky_retry_factor": 1e6,
        "fix_noise": None,
        "fix_noise_epochs": 3,
        "width": 256,
        "dim_embedding": 256,
        "enc_same": False,
        "num_heads": 8,
        "num_layers": 6,
        "unet_channels": (64,) * 6,
        "unet_strides": (1,) + (2,) * 5,
        "conv_channels": 64,
        "encoder_scales": None,
        "fullconvgnp_kernel_factor": 2,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
    }

    # Setup data generators for training and for evaluation.
    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**6 if args.train_fast else 2**14,
        num_tasks_cv=2**6 if args.evaluate_fast else 2**12,
        num_tasks_eval=2**6 if args.evaluate_fast else 2**12,
        device=device,
    )

    # Apply defaults for the number of epochs and the learning rate. The experiment
    # is allowed to adjust these.
    args.epochs = args.epochs or config["default"]["epochs"] or 100
    args.rate = args.rate or config["default"]["rate"] or 3e-4
    args.also_ar = args.also_ar or config["default"]["also_ar"]

    # Check if a run has completed.
    if args.check_completed:
        if os.path.exists(wd.file("model-last.torch")):
            d = torch.load(wd.file("model-last.torch"), map_location="cpu")
            if d["epoch"] >= args.epochs - 1:
                out.out("Completed!")
                sys.exit(0)
        out.out("Not completed.")
        sys.exit(1)

    # Set the regularisation based on the experiment settings.
    B.epsilon = config["epsilon"]
    B.cholesky_retry_factor = config["cholesky_retry_factor"]

    if "model" in config:
        # See if the experiment constructed the particular flavour of the model already.
        model = config["model"]
    else:
        # Construct the model.
        if args.model == "cnp":
            model = nps.construct_gnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="het",
                transform=config["transform"],
            )
        elif args.model == "gnp":
            model = nps.construct_gnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="lowrank",
                num_basis_functions=config["num_basis_functions"],
                transform=config["transform"],
            )
        elif args.model == "np":
            model = nps.construct_gnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="het",
                dim_lv=config["dim_embedding"],
                transform=config["transform"],
            )
        elif args.model == "acnp":
            model = nps.construct_agnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_heads=config["num_heads"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="het",
                transform=config["transform"],
            )
        elif args.model == "agnp":
            model = nps.construct_agnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_heads=config["num_heads"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="lowrank",
                num_basis_functions=config["num_basis_functions"],
                transform=config["transform"],
            )
        elif args.model == "anp":
            model = nps.construct_agnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_heads=config["num_heads"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="het",
                dim_lv=config["dim_embedding"],
                transform=config["transform"],
            )
        elif args.model == "convcnp":
            model = nps.construct_convgnp(
                points_per_unit=config["points_per_unit"],
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                likelihood="het",
                conv_arch=args.arch,
                unet_channels=config["unet_channels"],
                unet_strides=config["unet_strides"],
                conv_channels=config["conv_channels"],
                conv_layers=config["num_layers"],
                conv_receptive_field=config["conv_receptive_field"],
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        elif args.model == "convgnp":
            model = nps.construct_convgnp(
                points_per_unit=config["points_per_unit"],
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                likelihood="lowrank",
                conv_arch=args.arch,
                unet_channels=config["unet_channels"],
                unet_strides=config["unet_strides"],
                conv_channels=config["conv_channels"],
                conv_layers=config["num_layers"],
                conv_receptive_field=config["conv_receptive_field"],
                num_basis_functions=config["num_basis_functions"],
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        elif args.model == "convnp":
            if config["dim_x"] == 2:
                # Reduce the number of channels in the conv. architectures by a factor
                # $\sqrt(2)$. This keeps the runtime in check and reduces the parameters
                # of the ConvNP to the number of parameters of the ConvCNP.
                config["unet_channels"] = tuple(
                    int(c / 2**0.5) for c in config["unet_channels"]
                )
                config["dws_channels"] = int(config["dws_channels"] / 2**0.5)
            model = nps.construct_convgnp(
                points_per_unit=config["points_per_unit"],
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                likelihood="het",
                conv_arch=args.arch,
                unet_channels=config["unet_channels"],
                unet_strides=config["unet_strides"],
                conv_channels=config["conv_channels"],
                conv_layers=config["num_layers"],
                conv_receptive_field=config["conv_receptive_field"],
                dim_lv=16,
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        elif args.model == "fullconvgnp":
            model = nps.construct_fullconvgnp(
                points_per_unit=config["points_per_unit"],
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                conv_arch=args.arch,
                unet_channels=config["unet_channels"],
                unet_strides=config["unet_strides"],
                conv_channels=config["conv_channels"],
                conv_layers=config["num_layers"],
                conv_receptive_field=config["conv_receptive_field"],
                kernel_factor=config["fullconvgnp_kernel_factor"],
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        else:
            raise ValueError(f'Invalid model "{args.model}".')

    # Settings specific for the model:
    if config["fix_noise"] is None:
        if args.model in {"np", "anp", "convnp"}:
            config["fix_noise"] = True
        else:
            config["fix_noise"] = False

    # Ensure that the model is on the GPU and print the setup.
    model = model.to(device)
    if not args.load:
        out.kv(
            "Arguments",
            {
                attr: getattr(args, attr)
                for attr in args.__dir__()
                if not attr.startswith("_")
            },
        )
        out.kv(
            "Config", {k: "<custom>" if k == "model" else v for k, v in config.items()}
        )
        out.kv("Number of parameters", nps.num_params(model))

    # Perform evaluation.
    if args.evaluate_last:
        name = "model-last.torch"
    else:
        name = "model-best.torch"
    model.load_state_dict(torch.load(wd.file(name), map_location=device)["weights"])
    return model, args.model, args.dim_x


import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import trieste

from trieste.objectives.single_objectives import HARTMANN_3_MINIMIZER, HARTMANN_3_SEARCH_SPACE, hartmann_3
from trieste.objectives.single_objectives import HARTMANN_6_MINIMIZER, HARTMANN_6_SEARCH_SPACE, hartmann_6
from trieste.objectives.single_objectives import ACKLEY_5_MINIMIZER, ACKLEY_5_SEARCH_SPACE, ackley_5
from trieste.objectives.utils import mk_observer
from trieste.space import Box

from utils import plot_regret


def get_scaled_objective(objective, mean, variance):
    def f(x):
        x = (x + 1.) / 2.
        return (objective(x) - mean) / (variance ** 0.5)
    return f


def stats(problem):
    objective = OBJECTIVES[problem]
    search_space = SEARCH_SPACES[problem]
    observer = mk_observer(objective)
    num_initial_points = 10_000
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)
    return (
        tf.reduce_mean(initial_data.observations),
        tf.math.reduce_variance(initial_data.observations),
    )

import math

import tensorflow as tf

from trieste.space import Box
from trieste.types import TensorType


def rastrigin_4(x: TensorType) -> TensorType:
    r"""
    The 4-dimensional Rastrigin function is defined by:

    ::math:
        f(x) = A + \sum_{d=1}^4 (x_d^2 - A \cos(2 * \pi x_d)),

    where :math:`A=10`.

    This function has a global minimum at :math:`x=0` with :math:`f(x)=-30`.
    
    See Rastrigin, L. A. “Systems of extremal control.” Mir, Moscow (1974) for
    more details.

    Note that we rescale the original problem, which is typically defined
    over `[-5.12, 5.12]^4`.

    :param x: The points at which to evaluate the function, with shape [..., 4].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 4))])
    x = (2.0 * x - 1.0) * 5.12  # rescale from [0, 1]^4 to [-5.12, 5.12]^4
    A = 10.0
    r = tf.reduce_sum(x ** 2 - A * tf.math.cos(2 * math.pi * x), axis=-1, keepdims=True)
    y = A + r
    return y


RASTRIGIN_4_MINIMIZER = tf.ones((1, 4), tf.float64) * 0.5
"""
The global minimizer for the :func:`rastrigin_4` function, with shape [1, 4] and
dtype float64.
"""


RASTRIGIN_4_MINIMUM = tf.constant([-30.], tf.float64)
"""
The global minimum for the :func:`rastrigin_4` function, with shape [1] and dtype
float64.
"""


RASTRIGIN_4_SEARCH_SPACE = Box([0.0], [1.0]) ** 4
""" The search space for the :func:`rastrigin_4` function. """

OBJECTIVES = {
    "hartmann3": get_scaled_objective(hartmann_3, -0.93557, 0.901356),
    "hartmann6": get_scaled_objective(hartmann_6, -0.2588, 0.1455),
    "ackley5": get_scaled_objective(ackley_5, 20.9778, 0.63444),
    "rastrigin4": get_scaled_objective(rastrigin_4, 44.00, 410.87)
}

SEARCH_SPACES = {
    "hartmann3": Box([-1.] * 3, [1.0] * 3),
    "hartmann6": Box([-1.] * 6, [1.0] * 6),
    "ackley5": Box([-1.] * 5, [1.0] * 5),
    "rastrigin4": Box([-1.] * 4, [1.0] * 4),
}

MINIMA = {
    "hartmann3": OBJECTIVES["hartmann3"](2 * (HARTMANN_3_MINIMIZER - 0.5)),
    "hartmann6": OBJECTIVES["hartmann6"](2 * (HARTMANN_6_MINIMIZER - 0.5)),
    "ackley5": OBJECTIVES["ackley5"](2 * (ACKLEY_5_MINIMIZER - 0.5)),
    "rastrigin4": OBJECTIVES["rastrigin4"](2 * (RASTRIGIN_4_MINIMIZER - 0.5)),
}

from typing import Tuple
import tensorflow as tf
import numpy as np

from trieste.models.interfaces import TrainableProbabilisticModel
from trieste.data import Dataset
from trieste.types import TensorType
from gpflow.experimental.check_shapes import check_shapes


def f32(x):
    return tf.cast(x, dtype=tf.float32)

def f64(x):
    return tf.cast(x, dtype=tf.float64)


class NeuralProcessTriesteWrapper(TrainableProbabilisticModel):
    def __init__(self, model, initial_dataset: Dataset):
        self._model = model
        # self._input_dim = input_dim
        self._num_initial = initial_dataset.query_points.shape[0]
        self._input_dim = initial_dataset.query_points.shape[-1]
        self._X = tf.Variable(
            f32(initial_dataset.query_points),
            trainable=False,
            shape=[None, self._input_dim],
            dtype=tf.float32
        )
        self._Y = tf.Variable(
            f32(initial_dataset.observations),
            trainable=False,
            shape=[None, 1],
            dtype=tf.float32
        )

    @check_shapes(
        "query_points: [N, D]",
        "return[0]: [N, 1]",
        "return[1]: [N, 1]",
    )
    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        def _tr(t):
            return tf.transpose(t, [0, 2, 1])

        mean, var, samples, _ = nps.predict(
            self._model,
            torch.Tensor(_tr(self._X[None]).numpy()),
            torch.Tensor(_tr(self._Y[None]).numpy()),
            torch.Tensor(_tr(query_points[None]).numpy()),
        )
        mean = tf.convert_to_tensor(mean.detach().numpy())
        var = tf.convert_to_tensor(var.detach().numpy())
        return _tr(mean)[0], _tr(mean)[1]

    @check_shapes(
        "query_points: [num_query, input_dim]",
        "return: [num_samples, num_query, 1]",
    )
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:

        def _tr(t):
            return tf.transpose(t, [0, 2, 1])

        mean, var, samples, _ = nps.predict(
            self._model,
            torch.Tensor(_tr(self._X[None]).numpy()),
            torch.Tensor(_tr(self._Y[None]).numpy()),
            torch.Tensor(_tr(query_points[None]).numpy()),
            num_samples=num_samples,
        )
        samples = tf.convert_to_tensor(samples.detach().numpy())
        return _tr(samples[:, 0, :, :])

    def update(self, dataset: Dataset) -> None:
        self._X.assign(f32(dataset.query_points))
        self._Y.assign(f32(dataset.observations))
        
    def optimize(self, dataset: Dataset) -> None:
        pass

    def log(self):
        print("Step:", tf.shape(self._X)[0] - self._num_initial)




def run(model, seed=1, method="anp", problem="hartmann6", num_search_space_samples=124, num_steps=75, output_dir="logs"):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    base_path = Path(output_dir)
    base_path.mkdir(exist_ok=True)

    date = datetime.datetime.now().strftime("%b%d_%H%M%S")
    experiment_name = f"{date}_{problem}-{method}-{seed}-Ns-{num_steps}-S-{num_search_space_samples}"
    (base_path / method / experiment_name).mkdir(exist_ok=True, parents=True)
    # summary_writer = tf.summary.create_file_writer(f"logs/tensorboards/{date}_seed_{seed}")

    objective = OBJECTIVES[problem]
    search_space = SEARCH_SPACES[problem]

    observer = mk_observer(objective)
    num_initial_points = 5
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    model = NeuralProcessTriesteWrapper(model, initial_data)
    num_query_points = 1
    acq_rule = trieste.acquisition.rule.DiscreteThompsonSampling(
        num_search_space_samples=num_search_space_samples,
        num_query_points=num_query_points,
    )

    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

    result = bo.optimize(
        num_steps, initial_data, model, acq_rule, track_state=False
    )
    final_dataset = result.try_get_final_dataset()

    fig, ax = plt.subplots()
    regret = final_dataset.observations - MINIMA[problem]
    min_idx = tf.squeeze(tf.argmin(regret, axis=0))
    plot_regret(regret.numpy(), ax, num_init=num_initial_points, idx_best=min_idx)
    plt.savefig(
        str(base_path / method / experiment_name / f"{experiment_name}_regret.png"),
        facecolor="white",
        transparent=False,
    )

    np.savez(
        str(base_path / method / experiment_name / "final_dataset.npz"),
        query_points=final_dataset.query_points,
        observations=final_dataset.observations,
    )



if __name__ == "__main__":
    model, method, dimx = get_model()
    print(method)
    print(dimx)
    for seed in range(5):
        print("seed", seed)
        run(model, seed=seed, method=method, problem="rastrigin4", num_steps=50)
        # run(model, seed=seed, method=method, problem=f"hartmann{str(dimx)}", num_steps=35 if dimx == 3 else 75)
