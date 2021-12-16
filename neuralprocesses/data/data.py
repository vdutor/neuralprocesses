import lab as B
import matrix
import numpy as np
import stheno

__all__ = ["GPGenerator"]


class GPGenerator:
    """GP generator.

    Args:
        dtype (dtype): Data type to generate.
        kernel (:class:`stheno.Kernel`, optional): Kernel of the GP. Defaults to an
            EQ kernel with length scale `0.25`.
        noise (float, optional): Observation noise. Defaults to `5e-2`.
        seed (int, optional): Seed. Defaults to `0`.
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        x_range (tuple[tuple[float, float]...], optional): Ranges of the inputs. Every
            range corresponds to a dimension of the input, which means that the number
            of ranges determine the dimensionality of the input. Defaults to
            `((-2, 2),)`.
        num_context_points (int or tuple[int, int], optional): A fixed number of context
            points or a lower and upper bound. Defaults to the range `(1, 50)`.
        num_target_points (int or tuple[int, int], optional): A fixed number of target
            points or a lower and upper bound. Defaults to the fixed number `50`.
        pred_logpdf (bool, optional): Also compute the logpdf of the target set given
            the context set under the true GP. Defaults to `True`.
        pred_logpdf_diag (bool, optional): Also compute the logpdf of the target set
            given the context set under the true diagonalised GP. Defaults to `True`.
        device (str, optional): Device on which to generate data. Defaults to `cpu`.
    """

    def __init__(
        self,
        dtype,
        kernel=stheno.EQ().stretch(0.25),
        noise=0.05 ** 2,
        seed=0,
        batch_size=16,
        num_tasks=2 ** 14,
        x_range=((-2, 2),),
        num_context_points=(1, 50),
        num_target_points=50,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        device="cpu",
    ):
        self.dtype = dtype
        # Derive the right floating and integral data types from `dtype`.
        self.float64 = B.promote_dtypes(dtype, np.float64)
        self.int64 = B.dtype_int(self.float64)
        self.device = device

        self.kernel = kernel
        self.noise = noise

        # The random state must be created on the right device.
        with B.on_device(self.device):
            self.state = B.create_random_state(dtype, seed)

        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_batches = num_tasks // batch_size
        if self.num_batches * batch_size != num_tasks:
            raise ValueError(
                f"Number of tasks {num_tasks} must be a multiple of "
                f"the batch size {batch_size}."
            )

        self.x_dim = len(x_range)
        # Contruct tensors for the bounds on the input range. These must be `flaot64`s.
        with B.on_device(self.device):
            lower = B.stack(*(B.cast(self.float64, l) for l, _ in x_range))[None, :]
            upper = B.stack(*(B.cast(self.float64, u) for _, u in x_range))[None, :]
            self.x_range = B.to_active_device(lower), B.to_active_device(upper)

        # Ensure that `num_context_points` and `num_target_points` are tuples of lower
        # bounds and upper bounds.
        if not isinstance(num_context_points, tuple):
            num_context_points = (num_context_points, num_context_points)
        if not isinstance(num_target_points, tuple):
            num_target_points = (num_target_points, num_target_points)
        self.num_context_points = num_context_points
        self.num_target_points = num_target_points

        self.pred_logpdf = pred_logpdf
        self.pred_logpdf_diag = pred_logpdf_diag

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A task, which is a dictionary with keys `xc`, `yc`, `xt`, and `yt`.
                Also possibly contains the keys `pred_logpdf` and `pred_logpdf_diag`.
        """
        with B.on_device(self.device):
            batch = {}

            # Sample number of context and target points.
            lower, upper = self.num_context_points
            self.state, num_context_points = B.randint(
                self.state, self.int64, lower=lower, upper=upper + 1
            )
            self.state, num_target_points = B.randint(
                self.state, self.int64, lower=lower, upper=upper + 1
            )

            # Sample inputs.
            self.state, rand = B.rand(
                self.state,
                self.float64,
                self.batch_size,
                int(num_context_points + num_target_points),
                self.x_dim,
            )
            lower, upper = self.x_range
            x = lower + rand * (upper - lower)

            # Construct prior. Cast `noise` before moving it to the active device,
            # because Python scalars will not be interpreted as tensors and hence will\
            # not be moved to the GPU.
            noise = B.to_active_device(B.cast(self.dtype, self.noise))
            f = stheno.GP(self.kernel)

            # Sample context and target set.
            self.state, y = f(x, noise).sample(self.state)
            xc = x[:, :num_context_points, :]
            yc = x[:, :num_context_points, :]
            xt = x[:, num_context_points:, :]
            yt = x[:, num_context_points:, :]

            # Compute predictive logpdfs.
            if self.pred_logpdf or self.pred_logpdf_diag:
                f_post = f | (f(xc, noise), yc)
                fdd = f_post(xt, noise)
            if self.pred_logpdf:
                batch["pred_logpdf"] = fdd.logpdf(yt)
            if self.pred_logpdf_diag:
                fdd_diag = stheno.Normal(fdd.mean, matrix.Diagonal(B.diag(fdd.var)))
                batch["pred_logpdf_diag"] = fdd_diag.logpdf(yt)

            # Put the data dimension last and convert to the right data type.
            batch["xc"] = B.cast(self.dtype, B.transpose(xc))
            batch["yc"] = B.cast(self.dtype, B.transpose(yc))
            batch["xt"] = B.cast(self.dtype, B.transpose(xt))
            batch["yt"] = B.cast(self.dtype, B.transpose(yt))

            return batch

    def epoch(self):
        """Construct a generator for an epoch.

        Returns:
            generator: Generator for an epoch.
        """

        def lazy_gen_batch():
            return self.generate_batch()

        return (lazy_gen_batch() for _ in range(self.num_batches))