import obspython as obs
from pynput import keyboard
from datetime import datetime

def script_description():
	return "Log the keyboard inputs along with the video"

def script_load(settings):
    global listener
    obs.obs_frontend_add_event_callback(on_event)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Script Loaded")

def script_unload():
    global listener
    listener.stop()

def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_path(props, "keylogger_path", "Key logging path :", obs.OBS_PATH_DIRECTORY, "", "")
    #S.obs_property_set_modified_callback(b, callback)
    return props

def script_update(settings):
    global path_name
    global current_frame 
    current_frame = 0
    path_name = obs.obs_data_get_string(settings, "keylogger_path")
    print("Path updated")

def on_event(event):
    global current_frame 
    global keys_down
    global log_file
    global path_name
    if event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED:
        keys_down = set()
        full_path = path_name+"/"+datetime.now().strftime("%Y-%m-%d-%H:%M:%S")+".csv"
        log_file = open(full_path, "w")
        log_file.write("FRAME,KEY,STATUS\n")
        print("Keylog started at "+full_path)
    elif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED:
        log_file.close()
        current_frame = 0
        print("Keylog stopped")

def script_tick(seconds):
    global current_frame 
    if obs.obs_frontend_recording_active():
        current_frame += 1

def on_press(key):
    global current_frame
    global keys_down
    global log_file
    if obs.obs_frontend_recording_active():
        if key not in keys_down:
            val = "{},{},DOWN\n".format(current_frame, key)
            log_file.write(val)
            log_file.flush()
            print(val)
        keys_down.add(key)

def on_release(key):
    global current_frame
    global keys_down
    global log_file
    if obs.obs_frontend_recording_active():
        keys_down.remove(key)
        val = "{},{},UP\r\n".format(current_frame, key)
        log_file.write(val)
        log_file.flush()
        print(val)
