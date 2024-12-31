from dataclasses import dataclass
from types import LambdaType

import torch
from torch import Tensor

perp_neg_f_sb = (1, 0.5, -0.606)
perp_neg_f_fsb = (1, 0.5, +0.967)
perp_neg_f_fs = (4, 0.5, -2.426)  # f_fs(1) = 0, a, b > 0
perp_neg_f_sf = (4, 0.5, -2.426)


def shifted_exponential_decay(a, b, c, r):
    return a * torch.exp(-b * r) + c


def shift_azimuth_deg(azimuth):
    # shift azimuth angle (in degrees), to [-180, 180]
    return (azimuth + 180) % 360 - 180


def perpendicular_component(x, y):
    # get the component of x that is perpendicular to y
    eps = torch.ones_like(x[:, 0, 0, 0]) * 1e-6
    return (
        x
        - (
            torch.mul(x, y).sum(dim=[1, 2, 3])
            / torch.maximum(torch.mul(y, y).sum(dim=[1, 2, 3]), eps)
        ).view(-1, 1, 1, 1)
        * y
    )


@dataclass
class DirectionConfig:
    name: str
    prompt: LambdaType
    negative_prompt: LambdaType
    condition: LambdaType


overhead_threshold: float = 60.0
front_threshold: float = 45.0
back_threshold: float = 45.0

directions = [
    DirectionConfig(
        "side",
        lambda s: f"{s}, side view",
        lambda s: s,
        lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
    ),
    DirectionConfig(
        "front",
        lambda s: f"{s}, front view",
        lambda s: s,
        lambda ele, azi, dis: (shift_azimuth_deg(azi) > -front_threshold)
        & (shift_azimuth_deg(azi) < front_threshold),
    ),
    DirectionConfig(
        "back",
        lambda s: f"{s}, back view",
        lambda s: s,
        lambda ele, azi, dis: (shift_azimuth_deg(azi) > 180 - back_threshold)
        | (shift_azimuth_deg(azi) < -180 + back_threshold),
    ),
    DirectionConfig(
        "overhead",
        lambda s: f"{s}, overhead view",
        lambda s: s,
        lambda ele, azi, dis: ele > overhead_threshold,
    ),
]

direction2idx = {d.name: i for i, d in enumerate(directions)}


def get_text_embeddings(
    text_embeddings_vd,
    uncond_text_embeddings_vd,
    elevation,
    azimuth,
    camera_distances,
    view_dependent_prompting: bool = True,
):
    batch_size = elevation.shape[0]

    if view_dependent_prompting:
        # Get direction
        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in directions:
            direction_idx[d.condition(elevation, azimuth, camera_distances)] = (
                direction2idx[d.name]
            )

        # Get text embeddings
        text_embeddings = text_embeddings_vd[direction_idx]  # type: ignore
        uncond_text_embeddings = uncond_text_embeddings_vd[direction_idx]  # type: ignore
    else:
        text_embeddings = text_embeddings.expand(batch_size, -1, -1)  # type: ignore
        uncond_text_embeddings = uncond_text_embeddings.expand(  # type: ignore
            batch_size, -1, -1
        )

    # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
    return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)


def get_text_embeddings_perp_neg(
    text_embeddings_vd,
    uncond_text_embeddings_vd,
    elevation,
    azimuth,
    camera_distances,
    view_dependent_prompting: bool = True,
):
    assert view_dependent_prompting, "Perp-Neg only works with view-dependent prompting"

    batch_size = elevation.shape[0]

    direction_idx = torch.zeros_like(elevation, dtype=torch.long)
    for d in directions:
        direction_idx[d.condition(elevation, azimuth, camera_distances)] = (
            direction2idx[d.name]
        )
    # 0 - side view
    # 1 - front view
    # 2 - back view
    # 3 - overhead view

    pos_text_embeddings = []
    neg_text_embeddings = []
    neg_guidance_weights = []
    uncond_text_embeddings = []

    side_emb = text_embeddings_vd[0]
    front_emb = text_embeddings_vd[1]
    back_emb = text_embeddings_vd[2]
    overhead_emb = text_embeddings_vd[3]

    for idx, ele, azi, dis in zip(direction_idx, elevation, azimuth, camera_distances):
        azi = shift_azimuth_deg(azi)  # to (-180, 180)
        uncond_text_embeddings.append(uncond_text_embeddings_vd[idx])  # should be ""
        if idx.item() == 3:  # overhead view
            pos_text_embeddings.append(overhead_emb)  # side view
            # dummy
            neg_text_embeddings += [
                uncond_text_embeddings_vd[idx],
                uncond_text_embeddings_vd[idx],
            ]
            neg_guidance_weights += [0.0, 0.0]
        else:  # interpolating views
            if torch.abs(azi) < 90:
                # front-side interpolation
                # 0 - complete side, 1 - complete front
                r_inter = 1 - torch.abs(azi) / 90
                pos_text_embeddings.append(
                    r_inter * front_emb + (1 - r_inter) * side_emb
                )
                neg_text_embeddings += [front_emb, side_emb]
                neg_guidance_weights += [
                    -shifted_exponential_decay(*perp_neg_f_fs, r_inter),
                    -shifted_exponential_decay(*perp_neg_f_sf, 1 - r_inter),
                ]
            else:
                # side-back interpolation
                # 0 - complete back, 1 - complete side
                r_inter = 2.0 - torch.abs(azi) / 90
                pos_text_embeddings.append(
                    r_inter * side_emb + (1 - r_inter) * back_emb
                )
                neg_text_embeddings += [side_emb, front_emb]
                neg_guidance_weights += [
                    -shifted_exponential_decay(*perp_neg_f_sb, r_inter),
                    -shifted_exponential_decay(*perp_neg_f_fsb, r_inter),
                ]

    text_embeddings = torch.cat(
        [
            torch.stack(pos_text_embeddings, dim=0),
            torch.stack(uncond_text_embeddings, dim=0),
            torch.stack(neg_text_embeddings, dim=0),
        ],
        dim=0,
    )

    return text_embeddings, torch.as_tensor(
        neg_guidance_weights, device=elevation.device
    ).reshape(batch_size, 2)
