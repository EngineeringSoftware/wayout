from . import static, dynamic

def _get_cpp_view_name(view):
    rank = view.rank()

    if not 0 < rank < 8:
        raise ValueError(f"View rank {rank} is not allowed")

    dtype_mapping = {
        pk.DataType.int16: "int16_t",
        pk.DataType.int32: "int32_t",
        pk.DataType.int64: "int64_t",
        pk.DataType.uint16: "uint16_t",
        pk.DataType.uint32: "uint32_t",
        pk.DataType.uint64: "uint64_t",
        pk.DataType.float: "float",
        pk.DataType.double: "double"
    }

    params = {}
    dtype = dtype_mapping[view.dtype]
    if "const" in view.array.__class__.__name__:
        dtype += " const"

    params["dtype"] = dtype + "*" * rank

    if view.trait != Trait.TraitDefault:
        params["trait"] = f"Kokkos::MemoryTraits<Kokkos::{lib.get_memory_trait(view.trait.value)}>"

    if view.layout != Layout.LayoutDefault:
        params["layout"] = f"Kokkos::{lib.get_layout(view.layout.value)}"

    if view.space != MemorySpace.MemorySpaceDefault:
        params["space"] = f"Kokkos::{lib.get_memory_space(view.space.value)}"

    if "space" not in params:
        params["space"] = "Kokkos::DefaultExecutionSpace::memory_space"

    params_ordered: List[str] = []
    params_ordered.append(params["dtype"])
    if "layout" in params:
        params_ordered.append(params["layout"])
    if "space" in params:
        # FIXME
        if params["space"] == "Kokkos::HostSpaceDevice":
            params_ordered.append("Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>")
        else:
            params_ordered.append(params["space"])
    if "trait" in params:
        params_ordered.append(params["trait"])

    cpp_type: str = "Kokkos::View<"
    cpp_type += ",".join(params_ordered) + ">"

    return cpp_type


def _from_handle(handle):
    # cast views
    params = type(handle).__name__.split("_")
    dtype = pk.DataType(getattr(lib, params[1]))
    layout_str = params[3] if params[2] == "const" else params[2]
    if "Layout" not in layout_str:
        layout = Layout.LayoutDefault
    else:
        layout = pk.Layout(getattr(lib, layout_str))
    space = pk.MemorySpace(handle.memory_space)
    return pk.View(handle.shape, dtype, layout=layout, space=space, array=handle)


try:
    # insert hooks if pykokkos is present
    import pykokkos as pk
    from pykokkos.interface import Layout, MemorySpace, Trait
    from pykokkos.bindings import kokkos as lib

    pk.View._handle = property(lambda v: v.array)
    pk.View._cpp_name = property(lambda v: _get_cpp_view_name(v))
    dynamic.register_autocast("KokkosView", _from_handle)


except ModuleNotFoundError:
    exit(1)
    pass
    

