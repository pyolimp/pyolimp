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
    tau: float = 2e-5
    Lambda: float = 65.0
    c_high: float = 1.0
    c_low: float = 1 - c_high
    gap: float = 0.001
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None


def _tv_prox(
    z: torch.Tensor, lambda_: float, num_iter: int = 10
) -> torch.Tensor:
    """Проксимальный оператор для анизотропного TV (метод двойственных переменных)."""
    p1 = torch.zeros_like(z[..., :-1])  # Горизонтальные разности
    p2 = torch.zeros_like(z[..., :-1, :])  # Вертикальные разности
    L = 0.25  # Шаг, согласно Beck & Teboulle

    div_p = torch.zeros_like(z)
    for _ in range(num_iter):
        # Градиент по отношению к первообразной переменной
        grad = div_p - z / lambda_

        # Обновление двойственных переменных
        grad_p1 = grad[..., :-1] - grad[..., 1:]
        grad_p1 *= L
        p1 += grad_p1
        p1 = torch.clamp(p1, -lambda_, lambda_, out=p1)

        grad_p2 = grad[..., :-1, :] - grad[..., 1:, :]
        grad_p2 *= L
        p2 += grad_p2
        p2 = torch.clamp(p2, -lambda_, lambda_, out=p2)

        div_p[:] = 0.0
        # Вычисляем дивергенцию p
        # Горизонтальная дивергенция
        div_p[..., :-1] += p1
        div_p[..., 1:] -= p1
        # Вертикальная дивергенция
        div_p[..., :-1, :] += p2
        div_p[..., 1:, :] -= p2

    x = z - lambda_ * div_p
    return x


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
    Универсальный алгоритм FISTA для минимизации функции F(x) = f(x) + g(x).

    :param fx: Функция для вычисления f(x) (гладкая часть).
    :param gx: Функция для вычисления g(x) (несглаженная часть).
    :param gradf: Функция для вычисления градиента f(x).
    :param proxg: Проксимальный оператор для g, т. е. proxg(z, lr) ≈ argminₓ { ½∥x - z∥² + lr * g(x) }.
    :param x0: Начальное приближение.
    :param lr: Шаг (learning rate).
    :param max_iter: Максимальное число итераций.
    :param gap: Критерий останова по изменению значения целевой функции.
    :param progress: Функция обратного вызова для отслеживания прогресса (значения от 0.0 до 1.0).
    :param debug: Словарь для хранения отладочной информации.
    :return: Найденное решение x.
    """
    y = x0.clone()
    x_prev = x0.clone()
    t = 1.0
    loss_steps = []

    for i in range(max_iter):
        if progress is not None:
            progress(i / max_iter)

        # Градиентный шаг: вычисляем grad f в точке y
        grad = gradf(y)
        z = y - lr * grad

        # Проксимальный шаг: применяем prox оператор для g
        x_new = proxg(z, lr)

        # FISTA-ускорение (момент)
        t_new = (1 + (1 + 4 * t**2) ** 0.5) / 2
        gamma = (t - 1) / t_new
        y_new = x_new + gamma * (x_new - x_prev)

        # Вычисляем значение целевой функции для отслеживания
        loss = fx(x_new) + gx(x_new)
        loss_steps.append(loss.item())

        if debug is not None:
            debug["loss_step"] = loss_steps
            debug["precomp"] = x_new.clone()

        # Критерий останова
        if i > 0 and abs(loss_steps[-2] - loss_steps[-1]) < gap:
            break

        # Обновляем переменные для следующей итерации
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
    Деконволюция изображения по методу Монтальто с использованием FISTA и TV-регуляризации.
    """
    theta = parameters.theta
    tau = parameters.tau
    Lambda = parameters.Lambda
    c_high, c_low = parameters.c_high, parameters.c_low

    # Начальное приближение (масштабированное изображение)
    t_init = image * (c_high - c_low) + c_low

    # Определяем гладкую часть f(x)
    def fx(x: torch.Tensor) -> torch.Tensor:
        e = fft_conv(x, psf) - image
        func_l2 = torch.linalg.norm(e.flatten())
        func_borders = torch.sum(
            torch.exp(-Lambda * x) + torch.exp(-Lambda * (1 - x))
        )
        return func_l2 + tau * func_borders

    # Градиент f(x) вычисляем с помощью autograd
    def gradf(x: torch.Tensor) -> torch.Tensor:
        x_temp = x.clone().detach().requires_grad_(True)
        f_val = fx(x_temp)
        f_val.backward()
        return x_temp.grad

    # Несглаженная часть g(x) = theta * TV(x)
    def gx(x: torch.Tensor) -> torch.Tensor:
        tv_h = torch.sum(torch.abs(x[..., :-1] - x[..., 1:]))
        tv_v = torch.sum(torch.abs(x[..., :-1, :] - x[..., 1:, :]))
        return theta * (tv_h + tv_v)

    # Проксимальный оператор для g: решает
    # proxg(z, lr) = argminₓ { ½∥x - z∥² + lr * theta * TV(x) }
    def proxg(z: torch.Tensor, step: float) -> torch.Tensor:
        return _tv_prox(z, step * theta)

    # Запускаем универсальный алгоритм FISTA
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
