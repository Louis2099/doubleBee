# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)

@staticmethod
def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data

class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction. This obeys the physics constraint on friction values. However, it may not always be
    essential for the application. Thus, the flag is set to ``False`` by default.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
        Afterwards, these materials are randomly assigned to the geometries of the asset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)


def apply_wheel_physx_material(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    robot_prim_path_template: str = "/World/envs/env_{}/Doublebee",
    static_friction: float = 1.2,
    dynamic_friction: float = 0.9,
    restitution: float = 0.0,
    friction_combine_mode: str = "multiply",
    restitution_combine_mode: str = "multiply",
):
    """Apply a PhysX material to wheel collision prims at startup.

    The USD robot may have wheel colliders bound to rendering materials instead of
    PhysX materials; this event creates a single PhysX material and binds it to
    the wheel collision meshes (Body1/Body2 under MeshInstance_* in rightWheel
    and leftWheel) for each environment so that wheel-ground friction is correct.

    Call this as a startup event (mode="startup") so it runs once after the scene
    is built. All parameters except :attr:`env` and :attr:`env_ids` can be passed
    via the EventTermCfg params dict.

    Args:
        env: The manager-based environment.
        env_ids: Environment indices to update; if None, all envs are updated
            (typical for startup).
        robot_prim_path_template: Format string for the robot root prim path per
            env, with one "{}" for the env index (e.g. "/World/envs/env_{}/Doublebee").
        static_friction: Static friction coefficient for the wheel material.
        dynamic_friction: Dynamic friction coefficient for the wheel material.
        restitution: Restitution (bounciness) for the wheel material.
        friction_combine_mode: How to combine with other body's friction
            ("average", "min", "multiply", "max").
        restitution_combine_mode: How to combine restitution ("average", "min",
            "multiply", "max").
    """
    try:
        import omni.usd
        from pxr import UsdPhysics, UsdShade, PhysxSchema  # type: ignore[import-untyped]
    except ImportError as e:
        carb.log_warn(
            f"[apply_wheel_physx_material] Skipping wheel PhysX material: missing dependency ({e})."
        )
        return

    num_envs = env.scene.num_envs
    if env_ids is None:
        env_indices = list(range(num_envs))
    else:
        if isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.cpu().tolist()
        else:
            env_indices = list(env_ids)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_warn("[apply_wheel_physx_material] No USD stage; skipping.")
        return

    # Create a prim for the physics material (USD has no UsdPhysics.Material; use prim + MaterialAPI)
    mat_prim_path = "/World/PhysicsMaterials/WheelMaterial"
    material_prim = stage.DefinePrim(mat_prim_path, "Material")
    if not material_prim:
        carb.log_warn("[apply_wheel_physx_material] Failed to define material prim; skipping.")
        return

    # Apply UsdPhysics.MaterialAPI and set friction/restitution (API exists in pxr.UsdPhysics)
    mat_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    mat_api.CreateStaticFrictionAttr().Set(static_friction)
    mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    mat_api.CreateRestitutionAttr().Set(restitution)
    try:
        physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        physx_api.CreateFrictionCombineModeAttr().Set(friction_combine_mode)
        physx_api.CreateRestitutionCombineModeAttr().Set(restitution_combine_mode)
    except Exception:
        pass

    # Wheel collision prim paths relative to robot root (from Doublebee USD structure)
    wheel_collision_suffixes = [
        "rightWheel/MeshInstance_242/Body1",
        "rightWheel/MeshInstance_243/Body2",
        "leftWheel/MeshInstance_245/Body1",
        "leftWheel/MeshInstance_246/Body2",
    ]

    # Wrap the material prim as a UsdShade.Material for binding
    material = UsdShade.Material(material_prim)

    bound_count = 0
    for i in env_indices:
        i = int(i)
        robot_path = robot_prim_path_template.format(i)
        for suffix in wheel_collision_suffixes:
            prim_path = f"{robot_path}/{suffix}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                # Use UsdShade.MaterialBindingAPI (not UsdPhysics) for binding materials
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                if binding_api:
                    # Bind the material to this collision prim
                    binding_api.Bind(material)
                    bound_count += 1

    if bound_count > 0:
        carb.log_info(
            f"[apply_wheel_physx_material] Bound '{mat_prim_path}' to {bound_count} wheel collision prims."
        )


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    # Transform from patch-relative coordinates to world coordinates using env_origins
    # (env_origins are at the center of each terrain patch)
    positions += env.scene.env_origins[env_ids]
    
    #NOTE: Add height offset to prevent robot submersion
    # The terrain patches are at the terrain surface (Z ≈ 0), but the robot's root link
    # is at the bottom of the robot. We need to raise it so the robot body sits on the surface.
    # Offset accounts for: wheel radius + body clearance + safety margin
    # Center of mass is at 0.1m above root link, so 0.1m offset should position robot correctly
    robot_height_offset = 0.30  # 10cm - adjust if needed based on actual robot dimensions
    positions[:, 2] += robot_height_offset

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # Set initial velocities to zero (no random sampling)
    velocities = torch.zeros((len(env_ids), 6), device=asset.device, dtype=asset.data.default_root_state.dtype)

    # set into the physics simulation
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_terrain_aligned(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    align_axis: str = "x",
):
    """Reset the asset root state with aligned start/end positions for play mode.
    
    This function samples a start position from "init_pos" patches and a target position from "target" patches,
    ensuring they are aligned on the same X or Y axis. The robot is then oriented to face directly toward the target.
    
    This is designed for inference/play mode to provide consistent, aligned initialization for better visualization
    and evaluation.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        pose_range: Dictionary of pose ranges (only roll/pitch are used, yaw is computed from target direction).
        velocity_range: Dictionary of velocity ranges (not used, velocities set to zero).
        asset_cfg: Configuration for the asset to reset.
        align_axis: Which axis to align on ("x" or "y"). Defaults to "x".
            - "x": Start and end have same X coordinate (robot moves along Y axis)
            - "y": Start and end have same Y coordinate (robot moves along X axis)
    
    Raises:
        ValueError: If required terrain patches are not found.
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain_aligned' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )
    
    # obtain target patches
    target_positions: torch.Tensor = terrain.flat_patches.get("target")
    if target_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain_aligned' requires valid flat patches under 'target'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # Get terrain info for each environment
    terrain_levels = terrain.terrain_levels[env_ids]  # [len(env_ids)]
    terrain_types = terrain.terrain_types[env_ids]    # [len(env_ids)]
    env_origins = env.scene.env_origins[env_ids]      # [len(env_ids), 3]
    
    # Initialize position and orientation arrays
    positions = torch.zeros(len(env_ids), 3, device=env.device)
    orientations = torch.zeros(len(env_ids), 4, device=env.device)
    
    # Process each environment individually to ensure alignment
    for i, env_idx in enumerate(env_ids):
        level = terrain_levels[i].item()
        ttype = terrain_types[i].item()
        env_origin = env_origins[i, :]  # [3]
        
        # Get available patches for this environment's terrain type
        init_patches = valid_positions[level, ttype, :, :]  # [num_init_patches, 3]
        target_patches = target_positions[level, ttype, :, :]  # [num_target_patches, 3]
        
        num_init = init_patches.shape[0]
        num_target = target_patches.shape[0]
        
        if num_init == 0 or num_target == 0:
            # Fallback: use random position if no patches available
            if num_init > 0:
                patch_idx = int(torch.randint(0, num_init, (1,), device=env.device).item())
                positions[i, :] = init_patches[patch_idx, :] + env_origin
            else:
                positions[i, :] = env_origin
            # Default orientation (no rotation)
            orientations[i, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            continue
        
        # Sample a random start position
        init_idx = int(torch.randint(0, num_init, (1,), device=env.device).item())
        start_pos_relative = init_patches[init_idx, :]  # [3]
        start_pos_world = start_pos_relative + env_origin  # [3]
        
        # Find target patches that align with start position on the specified axis
        if align_axis == "x":
            # Align on X axis: find targets with same X coordinate (within tolerance)
            start_x = start_pos_world[0].item()
            target_x_coords = target_patches[:, 0] + env_origin[0]  # [num_target_patches]
            # Find targets within 0.5m of start X coordinate
            aligned_mask = torch.abs(target_x_coords - start_x) < 0.5
            aligned_targets = target_patches[aligned_mask, :]
        else:  # align_axis == "y"
            # Align on Y axis: find targets with same Y coordinate (within tolerance)
            start_y = start_pos_world[1].item()
            target_y_coords = target_patches[:, 1] + env_origin[1]  # [num_target_patches]
            # Find targets within 0.5m of start Y coordinate
            aligned_mask = torch.abs(target_y_coords - start_y) < 0.5
            aligned_targets = target_patches[aligned_mask, :]
        
        # If no aligned targets found, use all targets (fallback)
        if aligned_targets.shape[0] == 0:
            aligned_targets = target_patches
        
        # Sample a random aligned target
        target_idx = int(torch.randint(0, aligned_targets.shape[0], (1,), device=env.device).item())
        target_pos_relative = aligned_targets[target_idx, :]  # [3]
        target_pos_world = target_pos_relative + env_origin  # [3]
        
        # Force exact alignment on the specified axis
        if align_axis == "x":
            target_pos_world[0] = start_pos_world[0]
        else:  # align_axis == "y"
            target_pos_world[1] = start_pos_world[1]
        
        # Add height offset to target (for visualization)
        target_pos_world[2] += 0.3
        
        # Store the aligned target position in the command manager if available
        # This will be used by TerrainTargetDirectionCommand
        # NOTE: This is set AFTER command manager resamples, so it will overwrite any random target
        if hasattr(env, "command_manager") and hasattr(env.command_manager, "_terms"):
            if "base_velocity" in env.command_manager._terms:
                velocity_cmd = env.command_manager._terms["base_velocity"]
                if hasattr(velocity_cmd, "current_targets_w"):
                    # Set the aligned target (this happens after command manager resamples)
                    velocity_cmd.current_targets_w[env_idx, :] = target_pos_world
        
        # Set robot position
        robot_height_offset = 0.30
        positions[i, :] = start_pos_world
        positions[i, 2] += robot_height_offset
        
        # Compute orientation to face target
        # Direction from start to target in world frame
        direction_w = target_pos_world - positions[i, :]  # [3]
        direction_xy = direction_w[:2]  # [2] - XY only
        
        # Compute yaw angle to face target
        # NOTE: DoubleBee robot faces along +Y axis in body frame (not +X)
        # atan2(x, y) gives angle from +Y axis to vector [x, y]
        # This is the correct yaw angle for a robot that faces +Y
        yaw = torch.atan2(direction_xy[0], direction_xy[1])  # Angle from +Y axis in XY plane
        
        # Sample roll and pitch from pose_range (if provided)
        roll_range = pose_range.get("roll", (0.0, 0.0))
        pitch_range = pose_range.get("pitch", (0.0, 0.0))
        roll = math_utils.sample_uniform(
            torch.tensor(roll_range[0], device=env.device),
            torch.tensor(roll_range[1], device=env.device),
            (1,), device=env.device
        )[0]
        pitch = math_utils.sample_uniform(
            torch.tensor(pitch_range[0], device=env.device),
            torch.tensor(pitch_range[1], device=env.device),
            (1,), device=env.device
        )[0]
        
        # Convert to quaternion
        orientations[i, :] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    
    # Set initial velocities to zero
    velocities = torch.zeros((len(env_ids), 6), device=asset.device, dtype=asset.data.default_root_state.dtype)
    
    # Set into the physics simulation
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)



def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

