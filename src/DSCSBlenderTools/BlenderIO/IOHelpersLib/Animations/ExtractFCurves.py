import re
from typing import Dict, List

import bpy.types
import mathutils
import numpy as np

from ..Maths import lerp, euler_to_quat


BONE_PATTERN = re.compile(r"pose\.bones\[\"(.*)\"\]\.(.*)")


def collect_fcurves_from_action(action: bpy.types.Action) -> Dict[str, Dict[str, Dict[int, bpy.types.FCurve]]]:
    re_pattern = re.compile(r'.+\["(\w+)"\]\.(\w+)')
    result = {}

    for fcurve in action.fcurves:
        obj_name = ''
        obj_attr = ''
        data_path = fcurve.data_path
        if data_path.startswith('pose.bones'):
            obj_name, obj_attr = re_pattern.match(data_path).groups()
        elif data_path.startswith('DSCS_MaterialProperties.'):
            temp_list = data_path.split('.')
            obj_name = temp_list[0]
            obj_attr = '.'.join(temp_list[1:])

        if obj_name not in result:
            result[obj_name] = {}
        if obj_attr not in result[obj_name]:
            result[obj_name][obj_attr] = {}

        result[obj_name][obj_attr][fcurve.array_index] = fcurve

    return result


def collect_keyframe_from_fcurves(
        attr_name: str,
        fcurves_data: Dict[int, bpy.types.FCurve],
        frame_range: mathutils.Vector,
) -> Dict[int, List]:
    crv_st = int(frame_range.x)
    crv_en = int(frame_range.y) + 1
    default_values = {
        'rotation_quaternion': [1., 0., 0., 0.],
        'rotation_euler': [0., 0., 0.],
        'location': [0., 0., 0.],
        'scale': [1., 1., 1.],
    }
    def_val = default_values[attr_name]
    # Initialize output
    result = {each_frame: [] for each_frame in range(crv_st, crv_en)}
    # Create look up
    keyframe_look_up = {}
    for idx, fcurve in fcurves_data.items():
        for k in fcurve.keyframe_points:
            frame = k.co.x
            val = k.co.y
            if frame not in keyframe_look_up:
                keyframe_look_up[frame] = []
            keyframe_look_up[frame].insert(idx, val)

    for each_frame in range(crv_st, crv_en):
        if each_frame in keyframe_look_up:
            result[each_frame].extend(keyframe_look_up[each_frame])
        else:
            if len(keyframe_look_up) == 0:
                result[each_frame] = def_val
            elif len(keyframe_look_up) == 1:
                key_look_up = list(keyframe_look_up.keys())[-1]
                result[each_frame] = keyframe_look_up[key_look_up][:]
            else:
                for idx, fcurve in fcurves_data.items():
                    result[each_frame].insert(idx, fcurve.evaluate(each_frame))

    return result


def format_to_bone_fcurve_data(
        raw_fcurves_data: Dict[str, Dict[str, Dict[int, bpy.types.FCurve]]],
        bone_collection: Dict[str, bpy.types.Bone],
        frame_range: mathutils.Vector,
) -> Dict[str, Dict[str, Dict[int, List]]]:
    result = {}
    for obj_name, obj_data in raw_fcurves_data.items():
        if obj_name not in bone_collection:
            continue
        if obj_name not in result:
            result[obj_name] = {}
        bone = bone_collection[obj_name]
        rotation_mode = bone.rotation_mode
        for attr_name, fcurves_data in obj_data.items():
            keyframe_data = collect_keyframe_from_fcurves(attr_name, fcurves_data, frame_range)

            if rotation_mode != "QUATERNION" and attr_name == 'rotation_euler':
                keyframe_data = {k: [q for q in euler_to_quat(v, rotation_mode)] for k, v in keyframe_data.items()}
                attr_name = 'rotation_quaternion'

            result[obj_name][attr_name] = keyframe_data
        if 'rotation_quaternion' not in result[obj_name]:
            result[obj_name]['rotation_quaternion'] = {}
        if 'location' not in result[obj_name]:
            result[obj_name]['location'] = {}
        if 'scale' not in result[obj_name]:
            result[obj_name]['scale'] = {}

    return result


#############
# INTERFACE #
#############
def extract_fcurves(action):
    """
    Returns a dictionary of all fcurves from an action, grouped by datapath.
    """
    res = {}
    for fcurve in action.fcurves:
        data_path = fcurve.data_path
        if data_path not in res:
            res[data_path] = {}
        
        fcurve_data = res[data_path]
        array_index = fcurve.array_index
        fcurve_data[array_index] = fcurve
            
    return res


def bone_fcurves_from_fcurves(fcurves_dict):
    """
    Transfers all bone animation fcurves to a new dict.
    """
    res = {}
    for fc_dp, fcurve in list(fcurves_dict.items()):
        if fc_dp[:10] == "pose.bones":
            if fc_dp[11] == "[":
                bone_name, attribute_rna_path = get_bone_name_from_datapath(fc_dp)
                if bone_name not in res:
                    res[bone_name] = {}
                res[bone_name][attribute_rna_path] = fcurve
                del fcurves_dict[fc_dp]
    return res


def object_transforms_from_fcurves(fcurves_dict):
    """
    Transfers Object Transform fcurves to a new dict.
    """
    res = {}
    for fc_dp, fcurve in list(fcurves_dict.items()):
        if fc_dp in ["rotation_quaternion", "rotation_euler", "location", "scale"]:
            res[fc_dp] = fcurve
            del fcurves_dict[fc_dp]
    return res


def synchronize_keyframes(fcurves, fcurve_defaults, interpolation_function):
    """
    Returns a list of keyframe: list pairs. The keyframes are taken from all
    unique keyframes amongst the input keyframes. The lists are the same size
    as the fcurve_defaults, and contain the animation values at the appropriate
    keyframe if these values exist. If they do not exist, the value is
    interpolated at that keyframe from the interpolation_method. for any
    missing fcurves, the fcurve_default is used on all keyframes.
    
    Any animation channels with array indices that overflow fcurve_defaults
    are removed.
    """
    # First get every keyframe and set up return value
    all_frame_idxs = sorted(set().union(*[fc.keys() for fc in fcurves.values()]))
    res = {idx: [c for c in fcurve_defaults] for idx in all_frame_idxs}
    
    for (component_idx, framedata), default_value in zip(fcurves.items(), fcurve_defaults):
        if component_idx >= len(fcurve_defaults): 
            continue

        # Get all the frames at which the curve has data
        component_frame_idxs = list(framedata.keys())
        # Produce a function that will return the value for the frame, based on how many frames are available
        interp_method = produce_interpolation_method(component_frame_idxs, framedata, default_value, interpolation_function)
        component_frame_idxs = set(component_frame_idxs)
        
        for frame_idx in all_frame_idxs:
            if frame_idx not in component_frame_idxs:
                res[frame_idx][component_idx] = interp_method(frame_idx)
            else:
                res[frame_idx][component_idx] = framedata[frame_idx]

    return res


def synchronised_transforms_from_fcurves(fcurves):
    for fcurve_data in fcurves:
        if 'rotation_quaternion' in fcurve_data: fcurve_data['rotation_quaternion'] = synchronize_keyframes(fcurve_data['rotation_quaternion'], [1, 0, 0, 0], lerp)
        if 'rotation_euler'      in fcurve_data: fcurve_data['rotation_euler']      = synchronize_keyframes(fcurve_data['rotation_euler'],      [0, 0, 0],    lerp)
        if 'location'            in fcurve_data: fcurve_data['location']            = synchronize_keyframes(fcurve_data['location'],            [0, 0, 0],    lerp)
        if 'scale'               in fcurve_data: fcurve_data['scale']               = synchronize_keyframes(fcurve_data['scale'],               [1, 1, 1],    lerp)
    return fcurves


def synchronised_bone_data_from_fcurves(fcurves_dict):
    bone_fcurves = bone_fcurves_from_fcurves(fcurves_dict)
    synchronised_transforms_from_fcurves(bone_fcurves.values())
    return bone_fcurves


def synchronised_object_transforms_from_fcurves(fcurves_dict):
    obj_fcurves = object_transforms_from_fcurves(fcurves_dict)
    synchronised_transforms_from_fcurves(obj_fcurves.values())
    return obj_fcurves


def synchronised_quat_bone_data_from_fcurves(fcurves_dict, bones):
    bone_fcurves = synchronised_bone_data_from_fcurves(fcurves_dict)
    for bn, data in bone_fcurves.items():
        if bn in bones:
            bone = bones[bn]
            rotation_mode = bone.rotation_mode
            if rotation_mode != "QUATERNION" and 'rotation_euler' in data:
                data["rotation_quaternion"] = {k: euler_to_quat(e, rotation_mode) for k, e in data["rotation_euler"].items()}
        if 'rotation_euler' in data:
            del data['rotation_euler']
        if 'rotation_quaternion' not in data:
            data['rotation_quaternion'] = {}
    return bone_fcurves


def synchronised_quat_object_transforms_from_fcurves(fcurves_dict, obj):
    obj_fcurves = synchronised_object_transforms_from_fcurves(fcurves_dict)
    rotation_mode = obj.rotation_mode
    if rotation_mode != "QUATERNION" and 'rotation_euler' in obj_fcurves:
        obj_fcurves["rotation_quaternion"] = {k: euler_to_quat(e, rotation_mode) for k, e in obj_fcurves["rotation_euler"].items()}
    if 'rotation_euler' in obj_fcurves:
        del obj_fcurves['rotation_euler']
    if 'rotation_quaternion' not in obj_fcurves:
        obj_fcurves['rotation_quaternion'] = {}
    return obj_fcurves


#############
# UTILITIES #
#############
def get_bone_name_from_datapath(fcurve):
    bone_data = re.match(BONE_PATTERN, fcurve)
    return bone_data.group(1), bone_data.group(2)


def interpolate_keyframe(frame_idxs, frame_values, idx, interpolation_function):
    smaller_elements = [fidx for fidx in frame_idxs if idx >= fidx]
    next_smallest_frame = max(smaller_elements) if len(smaller_elements) else frame_idxs[0]
    larger_elements = [fidx for fidx in frame_idxs if idx <= fidx]
    next_largest_frame = min(larger_elements) if len(larger_elements) else frame_idxs[-1]

    if next_largest_frame == next_smallest_frame:
        t = 0  # Totally arbitrary, since the interpolation will be between two identical values
    else:
        t = (idx - next_smallest_frame) / (next_largest_frame - next_smallest_frame)

    # Should change lerp to the proper interpolation method
    min_value = frame_values[next_smallest_frame]
    max_value = frame_values[next_largest_frame]

    return interpolation_function(np.array(min_value), np.array(max_value), t)


def produce_interpolation_method(frame_idxs, frame_values, default_value, interpolation_function):
    """
    Returns an interpolation function dependant on the number of passed frames.
    """
    if len(frame_idxs) == 0:
        def interp_method(input_frame_idx):
            return default_value
    elif len(frame_idxs) == 1:
        value = frame_values[frame_idxs[0]]

        def interp_method(input_frame_idx):
            return value
    else:
        def interp_method(input_frame_idx):
            return interpolate_keyframe(frame_idxs, frame_values, input_frame_idx, interpolation_function)

    return interp_method
