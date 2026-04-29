"""Train activations to learn a 3D spherical-cap rotation.

This reproduces the spirit of the hemisphere/cap rotation figure: samples
``x`` lie on a 3D spherical cap and the target ``y`` is a fixed rotation of
that cap. The script trains the same one-hidden-layer model with different
activations and writes training curves plus final 3D scatter plots.

Examples:
    python benchmarks/train_cap_rotation.py --out results/cap_rotation
    python benchmarks/train_cap_rotation.py --activations relu colu
    JAX_PLATFORMS=cpu python benchmarks/train_cap_rotation.py --steps 3000
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jax_colu import colu, rcolu


ActivationName = str


def parse_axis(value: str) -> np.ndarray:
    axis = np.fromstring(value, sep=",", dtype=np.float32)
    if axis.shape != (3,):
        raise argparse.ArgumentTypeError("axis must be three comma-separated floats")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise argparse.ArgumentTypeError("axis must be nonzero")
    return axis / norm


def rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    x, y, z = axis.astype(np.float32)
    theta = np.deg2rad(angle_degrees).astype(np.float32)
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def random_cap(n: int, cap_angle_degrees: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_theta = np.deg2rad(cap_angle_degrees)
    min_z = np.cos(max_theta)
    z = rng.uniform(min_z, 1.0, size=n).astype(np.float32)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1).astype(
        np.float32
    )


def grid_cap(n_theta: int, n_phi: int, cap_angle_degrees: float) -> np.ndarray:
    theta = np.linspace(0.0, np.deg2rad(cap_angle_degrees), n_theta, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float32)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    return np.stack(
        [np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1
    ).reshape(-1, 3).astype(np.float32)


def make_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rot = rotation_matrix(args.rotation_axis, args.rotation_degrees)
    x_train = random_cap(args.train_points, args.cap_angle_degrees, args.seed)
    y_train = x_train @ rot.T
    x_eval = grid_cap(args.eval_theta, args.eval_phi, args.cap_angle_degrees)
    y_eval = x_eval @ rot.T
    return x_train, y_train.astype(np.float32), x_eval, y_eval.astype(np.float32)


def init_params(
    *,
    seed: int,
    hidden_width: int,
    activation: ActivationName,
    device: jax.Device,
) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    if activation in {"colu", "rcolu"}:
        # CoLU variants benefit from a slightly positive apex bias so early
        # gradients do not begin in the fully inactive cone branch.
        b1 = np.zeros((hidden_width,), dtype=np.float32)
        b1[: hidden_width // 4] = 0.05
    else:
        b1 = np.zeros((hidden_width,), dtype=np.float32)

    params_np = {
        "w1": (rng.normal(size=(3, hidden_width)).astype(np.float32) * np.sqrt(2.0 / 3.0)),
        "b1": b1,
        "w2": (rng.normal(size=(hidden_width, 3)).astype(np.float32) * np.sqrt(2.0 / hidden_width)),
        "b2": np.zeros((3,), dtype=np.float32),
    }
    return {key: jax.device_put(value, device) for key, value in params_np.items()}


def apply_activation(x: jax.Array, activation: ActivationName, cone_dim: int) -> jax.Array:
    if activation == "relu":
        return jax.nn.relu(x)
    if activation == "silu":
        return jax.nn.silu(x)
    if activation == "gelu":
        return jax.nn.gelu(x)
    if activation == "colu":
        return colu(x, dim=cone_dim)
    if activation == "rcolu":
        return rcolu(x, dim=cone_dim)
    raise ValueError(f"unknown activation {activation!r}")


def predict(
    params: dict[str, jax.Array],
    x: jax.Array,
    activation: ActivationName,
    cone_dim: int,
) -> jax.Array:
    h = x @ params["w1"] + params["b1"]
    h = apply_activation(h, activation, cone_dim)
    return h @ params["w2"] + params["b2"]


def mse_loss(
    params: dict[str, jax.Array],
    x: jax.Array,
    y: jax.Array,
    activation: ActivationName,
    cone_dim: int,
) -> jax.Array:
    pred = predict(params, x, activation, cone_dim)
    return jnp.mean(jnp.square(pred - y))


def init_adam(params: dict[str, jax.Array], device: jax.Device) -> dict[str, object]:
    zeros = {
        key: jax.device_put(np.zeros(value.shape, dtype=value.dtype), device)
        for key, value in params.items()
    }
    return {
        "count": jax.device_put(np.array(0, dtype=np.int32), device),
        "m": zeros,
        "v": zeros,
    }


def adam_update(
    params: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    state: dict[str, object],
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict[str, jax.Array], dict[str, object]]:
    count = state["count"] + 1
    m = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad),
        state["v"],
        grads,
    )
    count_f = count.astype(jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**count_f), m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**count_f), v)
    params = jax.tree_util.tree_map(
        lambda param, mh, vh: param - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, {"count": count, "m": m, "v": v}


def make_step(
    *,
    activation: ActivationName,
    cone_dim: int,
    lr: float,
    device: jax.Device,
):
    def step(params, state, x, y):
        loss, grads = jax.value_and_grad(mse_loss)(
            params, x, y, activation, cone_dim
        )
        params, state = adam_update(params, grads, state, lr=lr)
        return params, state, loss

    return jax.jit(step, device=device)


def make_eval(*, activation: ActivationName, cone_dim: int, device: jax.Device):
    def eval_loss(params, x, y):
        return mse_loss(params, x, y, activation, cone_dim)

    return jax.jit(eval_loss, device=device)


def train_one(
    *,
    activation: ActivationName,
    args: argparse.Namespace,
    device: jax.Device,
    x_train: jax.Array,
    y_train: jax.Array,
    x_eval: jax.Array,
    y_eval: jax.Array,
) -> tuple[list[dict[str, float | int | str]], dict[str, jax.Array]]:
    seed_offsets = {"relu": 11, "silu": 17, "gelu": 23, "colu": 31, "rcolu": 37}
    params = init_params(
        seed=args.seed + 1000 + seed_offsets[activation],
        hidden_width=args.hidden_width,
        activation=activation,
        device=device,
    )
    state = init_adam(params, device)
    step = make_step(
        activation=activation, cone_dim=args.cone_dim, lr=args.lr, device=device
    )
    eval_loss = make_eval(activation=activation, cone_dim=args.cone_dim, device=device)

    rows: list[dict[str, float | int | str]] = []
    for i in range(args.steps + 1):
        if i % args.log_every == 0:
            train_loss = float(jax.device_get(eval_loss(params, x_train, y_train)))
            test_loss = float(jax.device_get(eval_loss(params, x_eval, y_eval)))
            rows.append(
                {
                    "activation": activation,
                    "step": i,
                    "train_mse": train_loss,
                    "eval_mse": test_loss,
                }
            )
            print(
                f"{activation:>5s} step={i:5d} "
                f"train={train_loss:.6e} eval={test_loss:.6e}"
            )
        if i == args.steps:
            break
        params, state, _ = step(params, state, x_train, y_train)
    return rows, params


def write_curves(rows: Iterable[dict[str, float | int | str]], out_dir: Path) -> Path:
    path = out_dir / "cap_rotation_curves.csv"
    rows = list(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["activation", "step", "train_mse", "eval_mse"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_curves(rows: list[dict[str, float | int | str]], out_dir: Path) -> Path:
    path = out_dir / "cap_rotation_curves.pdf"
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for activation in sorted({str(row["activation"]) for row in rows}):
        series = [row for row in rows if row["activation"] == activation]
        ax.plot(
            [int(row["step"]) for row in series],
            [float(row["eval_mse"]) for row in series],
            label=activation,
            lw=2,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Eval MSE")
    ax.set_yscale("log")
    ax.set_title("3D cap rotation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_caps(
    *,
    params_by_activation: dict[str, dict[str, jax.Array]],
    args: argparse.Namespace,
    x_eval: jax.Array,
    y_eval: jax.Array,
    out_dir: Path,
    device: jax.Device,
) -> Path:
    path = out_dir / "cap_rotation_final.pdf"
    x_np = np.asarray(jax.device_get(x_eval))
    y_np = np.asarray(jax.device_get(y_eval))
    activations = list(params_by_activation)
    fig = plt.figure(figsize=(4.0 * (2 + len(activations)), 3.5))

    panels = [("input x", x_np), ("target y", y_np)]
    for activation, params in params_by_activation.items():
        pred_fn = jax.jit(
            lambda x, params=params, activation=activation: predict(
                params, x, activation, args.cone_dim
            ),
            device=device,
        )
        panels.append((f"{activation} prediction", np.asarray(jax.device_get(pred_fn(x_eval)))))

    for idx, (title, points) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), idx, projection="3d")
        colors = points[:, 2]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=8, cmap="cool")
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/cap_rotation", type=Path)
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["relu", "colu"],
        choices=["relu", "silu", "gelu", "colu", "rcolu"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--steps", default=2000, type=int)
    parser.add_argument("--log-every", default=25, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--hidden-width", default=64, type=int)
    parser.add_argument("--cone-dim", default=4, type=int)
    parser.add_argument("--train-points", default=2048, type=int)
    parser.add_argument("--eval-theta", default=28, type=int)
    parser.add_argument("--eval-phi", default=48, type=int)
    parser.add_argument("--cap-angle-degrees", default=85.0, type=float)
    parser.add_argument("--rotation-degrees", default=55.0, type=float)
    parser.add_argument("--rotation-axis", default=parse_axis("0.35,0.8,0.45"), type=parse_axis)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hidden_width % args.cone_dim != 0:
        raise SystemExit("--hidden-width must be divisible by --cone-dim")

    args.out.mkdir(parents=True, exist_ok=True)
    device = jax.devices(args.device)[0]
    x_train_np, y_train_np, x_eval_np, y_eval_np = make_data(args)
    x_train = jax.device_put(x_train_np, device)
    y_train = jax.device_put(y_train_np, device)
    x_eval = jax.device_put(x_eval_np, device)
    y_eval = jax.device_put(y_eval_np, device)

    all_rows: list[dict[str, float | int | str]] = []
    params_by_activation: dict[str, dict[str, jax.Array]] = {}
    for activation in args.activations:
        rows, params = train_one(
            activation=activation,
            args=args,
            device=device,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
        )
        all_rows.extend(rows)
        params_by_activation[activation] = params

    curves_csv = write_curves(all_rows, args.out)
    curves_pdf = plot_curves(all_rows, args.out)
    caps_pdf = plot_final_caps(
        params_by_activation=params_by_activation,
        args=args,
        x_eval=x_eval,
        y_eval=y_eval,
        out_dir=args.out,
        device=device,
    )
    print(f"Saved {curves_csv}")
    print(f"Saved {curves_pdf}")
    print(f"Saved {caps_pdf}")


if __name__ == "__main__":
    main()
