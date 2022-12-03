import argparse
from utils.trt_profile import query_profiles, load_engine, Profile


def is_dynamic_profile(profile: Profile) -> bool:
    input_dynamic = [b.isDynamic for b in profile.inputs]
    return any(input_dynamic)


def get_engine_info(engine_path: str) -> str:
    profiles = query_profiles(engine_path)
    engine = load_engine(engine_path)
    name = engine.name

    default_profile = profiles[0]
    num_bindings_per_profile = len(default_profile.inputs) + len(default_profile.outputs)
    num_input_bindings = len(default_profile.inputs)
    num_output_bindings = len(default_profile.outputs)
    input_names = [b.name for b in default_profile.inputs]
    output_names = [b.name for b in default_profile.outputs]
    input_binding_idx = [b.idx for b in default_profile.inputs]
    output_binding_idx = [b.idx for b in default_profile.outputs]

    ret_str = ""
    ret_str += f"Model name: {name}\n"
    ret_str += f"Number of bindings per profile: {num_bindings_per_profile}\n"
    ret_str += f"    num of inputs: {num_input_bindings}\n"
    for idx, name in zip(input_binding_idx, input_names):
        ret_str += f"        idx: {idx}, name: {name}\n"
    ret_str += f"    num of outputs: {num_output_bindings}\n"
    for idx, name in zip(output_binding_idx, output_names):
        ret_str += f"        idx: {idx}, name: {name}\n"

    ret_str += f"Number of optimization profiles: {len(profiles)}\n"
    for idx, p in enumerate(profiles):
        ret_str += f"    profile #{idx}, is_dynamic: {is_dynamic_profile(p)}\n"
        for input_binding in p.inputs:
            ret_str += f"        {input_binding}\n"
    return ret_str


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("engine", help="Path to tensorrt engine file")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    engine_path = args.engine
    info = get_engine_info(engine_path)
    print(info)
