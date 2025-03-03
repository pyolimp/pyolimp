from __future__ import annotations
import torch


def _enforce_constraint(tone_curves: torch.Tensor):
    """
    Enforce monotonically increasing constraint and
    between 0-1 constraint
    """
    # Reconstruct by integration and normalization
    g_sum = torch.sum(
        tone_curves,
        keepdim=True,
        dim=-1,
    )
    tone_curves = torch.cumsum(tone_curves, axis=-1) / g_sum
    return tone_curves


def apply_center(t1, t2, t3, t4, patches, a, b, c, d) -> torch.Tensor:
    I0 = torch.arange(t1.shape[0]).unsqueeze(1)
    I1 = torch.arange(t1.shape[1]).unsqueeze(1).unsqueeze(2)

    _, _, h, w, _ = patches.shape
    flat = patches.flatten(-3, -2)

    luts = [
        torch.stack(
            [
                t[I0, I1, 0, flat[:, :, :, 0]],
                t[I0, I1, 1, flat[:, :, :, 1]],
                t[I0, I1, 2, flat[:, :, :, 2]],
            ],
            dim=0,
        )
        for t in [t1, t2, t3, t4]
    ]
    return (
        (
            (
                b * c * luts[0]
                + d * c * luts[1]
                + a * b * luts[2]
                + a * d * luts[3]
            )
            / ((a + c) * (b + d))
        )
        .unfold(3, h, w)
        .permute(1, 2, 3, 4, 0)
    )


def _apply_border(t1, t2, patches, e, f) -> torch.Tensor:
    I0 = torch.arange(t1.shape[0]).unsqueeze(1)

    _, h, w, _ = patches.shape
    flat = patches.flatten(-3, -2)

    luts = [
        torch.stack(
            [
                t[I0, 0, flat[:, :, 0]],
                t[I0, 1, flat[:, :, 1]],
                t[I0, 2, flat[:, :, 2]],
            ],
            dim=0,
        )
        for t in [t1, t2]
    ]
    return (
        ((f * luts[0] + e * luts[1]) / (e + f))
        .unfold(2, h, w)
        .permute(1, 2, 3, 0)
    )


def _apply_corner(t, patch) -> torch.Tensor:
    h, w, _ = patch.shape
    flat = patch.flatten(-3, -2)
    return (
        torch.stack([t[0, flat[:, 0]], t[1, flat[:, 1]], t[2, flat[:, 2]]])
        .unfold(1, h, w)
        .permute(1, 2, 0)
    )


def apply_postprocess(
    image_int: torch.Tensor,
    tone_curves: torch.Tensor,
    grid_size: tuple[int, int],
    num_curves: int,
):
    grid_rows, grid_cols = grid_size

    # long for using as indices
    image_int = image_int.long()

    # reshape tone curves to grid
    t_curves_reshape = tone_curves.reshape(
        grid_rows, grid_cols, num_curves, 256
    )
    t_curves_reshape = t_curves_reshape.transpose(0, 1)

    output_image = torch.zeros_like(image_int, dtype=torch.float32)

    # each tile has its own lut from tone_curves
    tile_h = image_int.shape[0] // grid_rows
    tile_w = image_int.shape[1] // grid_cols

    # tile consists of 4 batches
    batch_h = tile_h // 2
    batch_w = tile_w // 2

    # right and bottom borders may have different sizes
    mod_h = image_int.shape[0] % grid_rows
    mod_w = image_int.shape[1] % grid_cols

    t1 = t_curves_reshape[0:-1, 0:-1]
    t2 = t_curves_reshape[1:, 0:-1]
    t3 = t_curves_reshape[0:-1, 1:]
    t4 = t_curves_reshape[1:, 1:]

    # center + left border + top border

    center_src_area = (
        image_int[: -(batch_h + mod_h), : -(batch_w + mod_w)]
        .reshape(
            grid_rows * 2 - 1, batch_h, grid_cols * 2 - 1, batch_w, num_curves
        )
        .permute(0, 2, 1, 3, 4)
    )
    center_dst_area = (
        output_image[: -(batch_h + mod_h), : -(batch_w + mod_w)]
        .reshape(
            grid_rows * 2 - 1, batch_h, grid_cols * 2 - 1, batch_w, num_curves
        )
        .permute(0, 2, 1, 3, 4)
    )

    w = batch_w
    h = batch_h
    ind_w = torch.arange(0, w)
    ind_h = torch.arange(0, h)
    rows, cols = torch.meshgrid(ind_h, ind_w, indexing="ij")

    # center right bottom patches
    a = rows.flatten()
    b = w + w - cols.flatten() - 1
    c = h + h - rows.flatten() - 1
    d = cols.flatten()

    patches = center_src_area[1:-1:2, 1:-1:2]
    center_dst_area[1:-1:2, 1:-1:2] = apply_center(
        t1, t2, t3, t4, patches, a, b, c, d
    )

    # left bottom patches
    a = rows.flatten()
    b = w - cols.flatten() - 1
    c = h + h - rows.flatten() - 1
    d = w + cols.flatten()

    # center
    patches = center_src_area[1:-1:2, 2::2]
    center_dst_area[1:-1:2, 2::2] = apply_center(
        t1, t2, t3, t4, patches, a, b, c, d
    )

    # left border
    patches = center_src_area[1:-1:2, 0]
    center_dst_area[1:-1:2, 0] = _apply_border(
        t1[0, :], t3[0, :], patches, a, c
    )

    # right top patches
    a = h + rows.flatten()
    b = w + w - cols.flatten() - 1
    c = h - rows.flatten() - 1
    d = cols.flatten()

    # center
    patches = center_src_area[2::2, 1:-1:2]
    center_dst_area[2::2, 1:-1:2] = apply_center(
        t1, t2, t3, t4, patches, a, b, c, d
    )

    # top border
    patches = center_src_area[0, 1:-1:2]
    center_dst_area[0, 1:-1:2] = _apply_border(
        t1[:, 0], t2[:, 0], patches, d, b
    )

    # top left patches
    a = h + rows.flatten()
    b = w - cols.flatten() - 1
    c = h - rows.flatten() - 1
    d = w + cols.flatten()

    # center
    patches = center_src_area[2::2, 2::2]
    center_dst_area[2::2, 2::2] = apply_center(
        t1, t2, t3, t4, patches, a, b, c, d
    )

    # left border
    patches = center_src_area[2::2, 0]
    center_dst_area[2::2, 0] = _apply_border(t1[0, :], t3[0, :], patches, a, c)

    # top border
    patches = center_src_area[0, 2::2]
    center_dst_area[0, 2::2] = _apply_border(t1[:, 0], t2[:, 0], patches, d, b)

    # top left corner
    center_dst_area[0, 0] = _apply_corner(t1[0, 0], center_src_area[0, 0])

    # right border
    w = batch_w + mod_w
    h = batch_h
    ind_w = torch.arange(0, w)
    ind_h = torch.arange(0, h)
    rows, cols = torch.meshgrid(ind_h, ind_w, indexing="ij")

    right_border_src = image_int[:-h, -w:].reshape(
        grid_rows * 2 - 1, h, w, num_curves
    )
    right_border_dst = output_image[:-h, -w:].reshape(
        grid_rows * 2 - 1, h, w, num_curves
    )

    # right bottom patches
    a = rows.flatten()
    b = w + w - cols.flatten() - 1
    c = h + h - rows.flatten() - 1
    d = cols.flatten()

    patches = right_border_src[1:-1:2]
    right_border_dst[1:-1:2] = _apply_border(
        t2[-1, :], t4[-1, :], patches, a, c
    )

    # right top patches
    a = h + rows.flatten()
    b = w + w - cols.flatten() - 1
    c = h - rows.flatten() - 1
    d = cols.flatten()

    patches = right_border_src[2::2]
    right_border_dst[2::2] = _apply_border(t2[-1, :], t4[-1, :], patches, a, c)

    # top right corner
    right_border_dst[0] = _apply_corner(t2[-1, 0], right_border_src[0])

    # bottom border
    w = batch_w
    h = batch_h + mod_h
    ind_w = torch.arange(0, w)
    ind_h = torch.arange(0, h)
    rows, cols = torch.meshgrid(ind_h, ind_w, indexing="ij")

    bottom_border_src = (
        image_int[-h:, :-w]
        .reshape(h, grid_cols * 2 - 1, w, 3)
        .permute(1, 0, 2, 3)
    )
    bottom_border_dst = (
        output_image[-h:, :-w]
        .reshape(h, grid_cols * 2 - 1, w, 3)
        .permute(1, 0, 2, 3)
    )
    # bottom right
    a = rows.flatten()
    b = w + w - cols.flatten() - 1
    c = h + h - rows.flatten() - 1
    d = cols.flatten()

    patches = bottom_border_src[1:-1:2]
    bottom_border_dst[1:-1:2] = _apply_border(
        t3[:, -1], t4[:, -1], patches, d, b
    )

    # bottom left
    a = rows.flatten()
    b = w - cols.flatten() - 1
    c = h + h - rows.flatten() - 1
    d = w + cols.flatten()

    patches = bottom_border_src[2::2]
    bottom_border_dst[2::2] = _apply_border(
        t3[:, -1], t4[:, -1], patches, d, b
    )

    # bot left corner
    bottom_border_dst[0] = _apply_corner(t3[0, -1], bottom_border_src[0])

    # bot right corner
    output_image[-h:, -w:] = _apply_corner(t4[-1, -1], image_int[-h:, -w:])

    return output_image.clip(0, 1)


def post_process(
    tone_curves: torch.Tensor,  # [6, 64, 3, 256]
    in_im_int: torch.Tensor,  # [6, 512, 512, 3]
    grid_size: tuple[int, int] = (8, 8),
    num_curves: int = 3,
    constraint_tc: bool = True,
):
    if constraint_tc:
        tone_curves = _enforce_constraint(tone_curves)

    interp_images = torch.vmap(
        lambda *args: apply_postprocess(
            *args, grid_size=grid_size, num_curves=num_curves
        )
    )

    return interp_images(in_im_int, tone_curves)
