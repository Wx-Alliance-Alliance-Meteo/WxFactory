from device import wx_cupy

import numpy

cpu_array_types = [numpy.ndarray]
gpu_array_types = []


def readable_size(num_bytes: float) -> str:
    units = ["B ", "kB", "MB", "GB", "TB", "PB", "EB"]

    def get_num():
        result = num_bytes
        for i in range(len(units) - 1):
            if result <= 2000:
                return result, units[i]
            result /= 1024.0
        return result, units[-1]

    num, u = get_num()
    # print(f"{num_bytes} -> {num}, {u}")

    return f"{num:6.1f} {u}"


def get_objet_mem(obj, name: str, verbose: bool):
    """Determine how much memory the given object uses with arrays (numpy or cupy)."""

    if obj is None:
        return

    if gpu_array_types == [] and wx_cupy.load_cupy():
        import cupy

        gpu_array_types.append(cupy.ndarray)

    total_cpu_mem = 0.0
    total_gpu_mem = 0.0
    gpu_arrays: dict[str, float] = {}
    cpu_arrays: dict[str, float] = {}

    if isinstance(obj, dict):
        items = obj.items()
    else:
        items = vars(obj).items()

    logged = []
    for k, v in items:
        if type(v) in cpu_array_types and v.flags["OWNDATA"] and id(v) not in logged:
            nbytes = v.nbytes
            cpu_arrays[k] = nbytes
            total_cpu_mem += nbytes
            logged.append(id(v))
        elif type(v) in gpu_array_types and v.flags["OWNDATA"] and id(v) not in logged:
            nbytes = v.nbytes
            gpu_arrays[k] = nbytes
            total_gpu_mem += nbytes
            logged.append(id(v))

    cpu_string = f" Total host memory:   {readable_size(total_cpu_mem)}\n"
    gpu_string = f" Total device memory: {readable_size(total_gpu_mem)}\n"
    if verbose:
        cpu_threshold = total_cpu_mem / 50.0
        gpu_threshold = total_gpu_mem / 50.0

        cpu_rem = 0.0
        cpu_num = 0
        for k, v in sorted(cpu_arrays.items(), key=lambda x: x[1], reverse=True):
            if v > cpu_threshold:
                cpu_string += f"  - {k:24s}: {readable_size(v)}\n"
            else:
                cpu_rem += v
                cpu_num += 1
        if cpu_num > 0:
            cpu_string += f"  - [{cpu_num:3d} others]:             {readable_size(cpu_rem)}"

        gpu_rem = 0.0
        gpu_num = 0
        for k, v in sorted(gpu_arrays.items(), key=lambda x: x[1], reverse=True):
            if v > gpu_threshold:
                gpu_string += f"  - {k:24s}: {readable_size(v)}\n"
            else:
                gpu_rem += v
                gpu_num += 1
        if gpu_num > 0:
            gpu_string += f"  - [{gpu_num:3d} others]:             {readable_size(gpu_rem)}"

    print(f"{name}\n" + cpu_string + gpu_string, flush=True)
