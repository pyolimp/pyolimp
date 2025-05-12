from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
from olimp.processing import fft_conv


class DebugInfo(TypedDict):
    loss_step: list[float]
    precomp: torch.Tensor


class MontaltoParameters(NamedTuple):
    lr: float = 20
    theta: float = 1e-6
    c_high: float = 1.0
    c_low: float = 1 - c_high
    gap: float = 0.001
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None


def _tv_prox(
    z: torch.Tensor, lambda_: float, lr: float, num_iter: int = 10
) -> torch.Tensor:
    """Proximal operator for anisotropic TV (method of dual functions)."""
    p1 = torch.zeros_like(z[..., :-1])  # Horizontal differences
    p2 = torch.zeros_like(z[..., :-1, :])  # Vertical differences

    div_p = torch.zeros_like(z)
    for _ in range(num_iter):
        # Gradient with respect to the primitive variable
        grad = div_p - z / lambda_

        # Updating Dual Variables
        grad_p1 = grad[..., :-1] - grad[..., 1:]
        grad_p1 *= 1 / lr
        p1 += grad_p1
        p1 = torch.clamp(p1, -lambda_, lambda_, out=p1)

        grad_p2 = grad[..., :-1, :] - grad[..., 1:, :]
        grad_p2 *= 1 / lr
        p2 += grad_p2
        p2 = torch.clamp(p2, -lambda_, lambda_, out=p2)

        div_p[:] = 0.0
        # Calculate the divergence p
        # Horizontal divergence
        div_p[..., :-1] += p1
        div_p[..., 1:] -= p1
        # Vertical divergence
        div_p[..., :-1, :] += p2
        div_p[..., 1:, :] -= p2

    x = z - lambda_ * div_p
    return x.clamp(0, 1)


def FISTA(
    fx: Callable[[torch.Tensor], torch.Tensor],
    gx: Callable[[torch.Tensor], torch.Tensor],
    gradf: Callable[[torch.Tensor], torch.Tensor],
    proxg: Callable[[torch.Tensor, float], torch.Tensor],
    x0: torch.Tensor,
    lr: float,
    max_iter: int = 5000,
    gap: float = 0.001,
    progress: Callable[[float], None] | None = None,
    debug: DebugInfo | None = None,
) -> torch.Tensor:
    """
    Universal FISTA algorithm for minimizing the function F(x) = f(x) + g(x).

    :param fx: Function to compute f(x) (smooth part).
    :param gx: Function to compute g(x) (non-smooth part).
    :param gradf: Function to compute the gradient of f(x).
    :param proxg: Proximal operator for g, i.e., proxg(z, lr) ≈ argminₓ { ½∥x - z∥² + lr * g(x) }.
    :param x0: Initial approximation.
    :param lr: Step size (learning rate).
    :param max_iter: Maximum number of iterations.
    :param gap: Stopping criterion based on the change in the objective function value.
    :param progress: Callback function for tracking progress (values from 0.0 to 1.0).
    :param debug: Dictionary for storing debugging information.
    :return: The computed solution x.
    """

    y = x0.clone()
    x_prev = x0.clone()
    t = 1.0
    loss_steps = []

    for i in range(max_iter):
        if progress is not None:
            progress(i / max_iter)

        # Gradient step: calculate grad f at point y
        grad = gradf(y)
        z = y - lr * grad

        # Proximal step: apply prox operator to g
        x_new = proxg(z, lr)

        # FISTA-acceleration (torque)
        t_new = (1 + (1 + 4 * t**2) ** 0.5) / 2
        gamma = (t - 1) / t_new
        y_new = x_new + gamma * (x_new - x_prev)

        # Calculate the value of the objective function for tracking
        loss = fx(x_new) + gx(x_new)
        loss_steps.append(loss.item())

        if debug is not None:
            debug["loss_step"] = loss_steps
            debug["precomp"] = x_new.clone()

        # Stopping Criteria
        if i > 0 and abs(loss_steps[-2] - loss_steps[-1]) < gap:
            break

        # Update variables for the next iteration
        x_prev = x_new.clone()
        y = y_new.clone()
        t = t_new

    if progress is not None:
        progress(1.0)

    return x_prev


def montalto(
    image: torch.Tensor,
    psf: torch.Tensor,
    parameters: MontaltoParameters = MontaltoParameters(),
) -> torch.Tensor:
    """
    Montalto image deconvolution using FISTA and TV regularization.
    """
    theta = parameters.theta
    c_high, c_low = parameters.c_high, parameters.c_low

    # Initial approximation (scaled image)
    t_init = image * (c_high - c_low) + c_low

    # Determine the smooth part of f(x)
    def fx(x: torch.Tensor) -> torch.Tensor:
        e = fft_conv(x, psf) - t_init
        func_l2 = torch.linalg.norm(e.flatten())
        return func_l2

    # Gradient f(x) is calculated using autogradd
    def gradf(x: torch.Tensor) -> torch.Tensor:
        x_temp = x.clone().detach().requires_grad_(True)
        f_val = fx(x_temp)
        f_val.backward()
        return x_temp.grad

    # Unsmoothed part g(x) = theta * TV(x)
    def gx(x: torch.Tensor) -> torch.Tensor:
        tv_h = torch.sum(torch.abs(x[..., :-1] - x[..., 1:]))
        tv_v = torch.sum(torch.abs(x[..., :-1, :] - x[..., 1:, :]))
        return theta * (tv_h + tv_v)

    # Proximal operator for g: solves
    # proxg(z, lr) = argminₓ { ½∥x - z∥² + lr * theta * TV(x) }
    def proxg(z: torch.Tensor, step: float) -> torch.Tensor:
        return _tv_prox(z, step * theta, step).clamp(0, 1)

    # Running the universal FISTA algorithm
    x_opt = FISTA(
        fx,
        gx,
        gradf,
        proxg,
        x0=t_init,
        lr=parameters.lr,
        max_iter=5000,
        gap=parameters.gap,
        progress=parameters.progress,
        debug=parameters.debug,
    )

    return x_opt


def _demo():
    from .._demo import demo

    def demo_montalto(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        return montalto(image, psf, MontaltoParameters(progress=progress))

    demo("Montalto (FISTA)", demo_montalto, mono=False)


if __name__ == "__main__":
    _demo()
