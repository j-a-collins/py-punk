from pydantic import BaseModel, Field


class TelemetryRow(BaseModel):
    """
    single time-step telemetry record.

    parameters
    ----------

    run_id : str
        identifier for a single driving run.
    frame_idx : int
        monotonic frame index within the run.
    t_wall_s : float
        wall-clock timestamp in seconds.
    t_run_s : float
        elapsed run time in seconds from episode start.
    player_in_vehicle : bool
        whether the player is currently inside a vehicle.
    vehicle_id : str | None
        identifier for the active vehicle, if available.
    pos_x : float
        world x position.
    pos_y : float
        world y position.
    pos_z : float
        world z position.
    vel_x : float
        world x velocity.
    vel_y : float
        world y velocity.
    vel_z : float
        world z velocity.
    speed_mps : float
        scalar speed in metres per second.
    yaw_rad : float
        yaw angle in radians.
    pitch_rad : float
        pitch angle in radians.
    roll_rad : float
        roll angle in radians.
    throttle_cmd : float
        applied throttle command in [0, 1].
    brake_cmd : float
        applied brake command in [0, 1].
    steer_cmd : float
        applied steering command in [-1, 1].
    manual_override : bool
        whether the controls are currently human-driven.
    collision_flag : bool
        whether a collision occurred at this time-step.
    episode_done : bool
        whether the episode terminated at this time-step.
    done_reason : str
        free-form termination reason.
    """

    run_id: str
    frame_idx: int = Field(ge=0)
    t_wall_s: float
    t_run_s: float = Field(ge=0.0)

    player_in_vehicle: bool
    vehicle_id: str | None = None

    pos_x: float
    pos_y: float
    pos_z: float

    vel_x: float
    vel_y: float
    vel_z: float
    speed_mps: float = Field(ge=0.0)

    yaw_rad: float
    pitch_rad: float
    roll_rad: float

    throttle_cmd: float = Field(ge=0.0, le=1.0)
    brake_cmd: float = Field(ge=0.0, le=1.0)
    steer_cmd: float = Field(ge=-1.0, le=1.0)

    manual_override: bool
    collision_flag: bool
    episode_done: bool
    done_reason: str = ""
