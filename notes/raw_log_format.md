# raw telemetry log format

the game-side logger should emit one JSON object per line (`.jsonl`).

each line should contain:

- run_id
- frame_idx
- t_wall_s
- t_run_s
- player_in_vehicle
- vehicle_id
- pos_x
- pos_y
- pos_z
- vel_x
- vel_y
- vel_z
- speed_mps
- yaw_rad
- pitch_rad
- roll_rad
- throttle_cmd
- brake_cmd
- steer_cmd
- manual_override
- collision_flag
- episode_done
- done_reason

notes:
- one line per simulation tick
- UTF-8
- newline-delimited JSON
- append-only during a run
- exactly one terminal row per completed run if possible