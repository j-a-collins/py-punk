local LOG_FILE = "telemetry_probe.jsonl"
local SAMPLE_PERIOD_S = 0.05

local run_id = "probe_" .. tostring(os.time())
local frame_idx = 0
local started_at = nil
local last_sample_at = nil

local last_pos_x = 0.0
local last_pos_y = 0.0
local last_pos_z = 0.0
local last_in_vehicle = false

local function bool_json(x)
    return x and "true" or "false"
end

local function str_json(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\"):gsub('"', '\\"')
    return '"' .. s .. '"'
end

local function write_line(line)
    local f = io.open(LOG_FILE, "a")
    if f == nil then
        print("[py_punk_probe] failed to open " .. LOG_FILE)
        return
    end
    f:write(line)
    f:write("\n")
    f:close()
end

local function make_row(now, t_run_s, in_vehicle, pos_x, pos_y, pos_z, episode_done, done_reason)
    return string.format(
        '{"run_id":%s,"frame_idx":%d,"t_wall_s":%.6f,"t_run_s":%.6f,' ..
        '"player_in_vehicle":%s,"vehicle_id":"","pos_x":%.6f,"pos_y":%.6f,"pos_z":%.6f,' ..
        '"vel_x":0.0,"vel_y":0.0,"vel_z":0.0,"speed_mps":0.0,' ..
        '"yaw_rad":0.0,"pitch_rad":0.0,"roll_rad":0.0,' ..
        '"throttle_cmd":0.0,"brake_cmd":0.0,"steer_cmd":0.0,' ..
        '"manual_override":true,"collision_flag":false,"episode_done":%s,"done_reason":%s}',
        str_json(run_id),
        frame_idx,
        now,
        t_run_s,
        bool_json(in_vehicle),
        pos_x,
        pos_y,
        pos_z,
        bool_json(episode_done),
        str_json(done_reason)
    )
end

local function ensure_log_cleared()
    local f = io.open(LOG_FILE, "w")
    if f ~= nil then
        f:write("")
        f:close()
    end
end

local function sample_once()
    local player = Game.GetPlayer()
    if player == nil then
        return
    end

    local vehicle = Game.GetMountedVehicle(player)
    local subject = vehicle or player
    local pos = subject:GetWorldPosition()
    if pos == nil then
        return
    end

    local now = os.clock()

    if started_at == nil then
        started_at = now
        last_sample_at = now - SAMPLE_PERIOD_S
        ensure_log_cleared()
        print("[py_punk_probe] started " .. run_id)
    end

    if now - last_sample_at < SAMPLE_PERIOD_S then
        return
    end

    last_sample_at = now
    last_pos_x = pos.x
    last_pos_y = pos.y
    last_pos_z = pos.z
    last_in_vehicle = vehicle ~= nil

    local row = make_row(
        now,
        now - started_at,
        last_in_vehicle,
        last_pos_x,
        last_pos_y,
        last_pos_z,
        false,
        ""
    )
    write_line(row)
    frame_idx = frame_idx + 1
end

registerForEvent("onInit", function()
    print("[py_punk_probe] loaded")
end)

registerForEvent("onUpdate", function(_)
    sample_once()
end)

registerForEvent("onShutdown", function()
    if started_at == nil then
        return
    end

    local now = os.clock()
    local row = make_row(
        now,
        now - started_at,
        last_in_vehicle,
        last_pos_x,
        last_pos_y,
        last_pos_z,
        true,
        "shutdown"
    )
    write_line(row)
    print("[py_punk_probe] wrote terminal row")
end)