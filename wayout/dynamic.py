import enum
import ctypes
import hashlib
import importlib
import os
import re
import typing
import sys

import pykokkos as pk
from pykokkos.interface import Layout, MemorySpace, Trait
from pykokkos.bindings import kokkos as lib

# benchmark info
total_build_time = 0
timer = pk.Timer()
generated_ctors = set()
generated_kernels = set()

class ptr:
    def __init__(self, val):
        self.val = val 


class char_ptr:
    def __init__(self, val):
        self.val = val 


def get_handle(v):
    if isinstance(v, pk.View):
        return v.array
    elif hasattr(v, "_handle"):
        return v._handle
    elif isinstance(v, ptr) or isinstance(v, char_ptr):
        return get_handle(v.val)
    else:
        return v


# mapping of operator to cpp name
class Operator(enum.Enum):
    ADD = "ADD_"
    SUB = "SUB_"
    GET_ITEM = "GET_ITEM_"
    SET_ITEM = "SET_ITEM_"
    CALL = "CALL_"
    DEREF = "DEREF_"


def get_view_name(view):
    shape = view.shape
    dtype = view.dtype.value
    space = view.space.value
    layout = view.layout.value
    trait = view.trait.value
    _prefix = "KokkosView"
    _space = lib.get_memory_space(space)
    _dtype = lib.get_dtype(dtype)
    _name = None
    if layout is not None:
        _layout = lib.get_layout(layout)
        # LayoutRight is the default
        if _layout != "LayoutRight":
            _name = "{}_{}_{}_{}".format(_prefix, _dtype, _layout, _space)
    if trait is not None:
        _trait = lib.get_memory_trait(trait)
        if _trait == "Unmanaged":
            raise ValueError("Use unmanaged_array() for the unmanaged view memory trait")
        _name = "{}_{}_{}_{}".format(_prefix, _dtype, _space, _trait)
    if _name is None:
        _name = "{}_{}_{}".format(_prefix, _dtype, _space)
    _name = "{}_{}".format(_name, len(shape))
    return _name


def get_subview_name(subview):
    dtype = subview.parent_view.dtype.value
    space = subview.parent_view.space.value
    _prefix = "KokkosView"
    _dtype = lib.get_dtype(dtype)
    _space = lib.get_memory_space(space)
    _unmanaged = lib.get_memory_trait(lib.Unmanaged)
    _name = "{}_{}_{}_{}_{}".format(_prefix, _dtype, _space, _unmanaged, subview.ndim)
    return _name


def get_cpp_view_name(view):
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


_cpp_types = {
    str: "std::string",
    int: "int",
    float: "double",
    bool: "bool",
    # use None as both class and actual value
    None: "void",
    type(None): "void",
}


def import_module(lib_path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, lib_path)
    if spec is None:
        raise ModuleNotFoundError
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)


def get_qualified_name(prefix, namespace, args, template_args):
    if namespace:
        prefix = f"{namespace}::{prefix}"

    cpp_args = []
    for arg in args:
        is_ptr = False
        if isinstance(arg, ptr):
            arg = arg.val
            is_ptr = True

        if type(arg) in _cpp_types:
            typename = _cpp_types[type(arg)]
        elif isinstance(arg, char_ptr):
            typename = "char *"
        elif isinstance(arg, pk.View) or isinstance(arg, pk.Subview):
            typename = get_cpp_view_name(arg)
        elif hasattr(arg, "_cpp_name"):
            typename = arg._cpp_name
        else:
            # fallback for edge cases (e.g enums)
            # raise TypeError(f"Invalid argument type {arg}!")
            typename = type(arg).__name__

        if is_ptr:
            typename += "*"
        cpp_args.append(typename)

    template_str = ""
    if template_args:
        template_str = f"<{get_cpp_name('', None, template_args)}>"

    return f"{prefix}{template_str}({','.join(cpp_args)})"


# generate argument casts (expect list of instances) 
def gen_arg_casts(buf, args):
    for i, _arg in enumerate(args):
        # cast to reference by default
        ref = " &"

        if isinstance(_arg, ptr) or isinstance(_arg, char_ptr):
            arg = _arg.val
            ref = " *"
        else:
            arg = _arg

        if type(arg) in _cpp_types:
            typename = _cpp_types[type(arg)]
        elif isinstance(arg, pk.View) or isinstance(arg, pk.Subview):
            typename = get_cpp_view_name(arg) + ref
        elif hasattr(arg, "_cpp_name"):
            typename = arg._cpp_name + ref
        # arg is enum
        elif arg.__module__ == "build._kernel_enums":
            namespace = arg.__doc__[:arg.__doc__.index("\n")]
            typename = type(arg).__name__    
            if namespace != "None":
                typename = f"{namespace}::" + typename
        else:
            raise TypeError("Unknown cast!", _arg)

        buf.append(f"{typename} a{i} = args[{i}].cast<{typename}>();")


# get cpp name of templated class/arg (expect list of instances) 
def get_cpp_name(prefix, namespace, template_args):
    if namespace:
        prefix = f"{namespace}::{prefix}"

    if template_args:
        # TODO: handle this more elegantly
        if len(template_args) == 1 and template_args[0] == []:
            return f"{prefix}<>"
        cpp_args = []
        for arg in template_args:
            # manual type override
            if isinstance(arg, str):
                typename = arg
            elif arg in _cpp_types:
                typename = _cpp_types[arg]
            elif isinstance(arg, pk.View) or isinstance(arg, pk.Subview):
                typename = get_cpp_view_name(arg)
            elif hasattr(arg, "_cpp_name"):
                if arg._handle is not None:
                    print("Warning: using instance object as template argument!", file=sys.stderr)
                typename = arg._cpp_name
            elif isinstance(arg, int):
                # cpp allows int as template parameter
                typename = str(arg)
            else:
                raise TypeError(f"Invalid template argument {arg}!")

            cpp_args.append(typename)

        return f"{prefix}<{','.join(cpp_args)}>"
    else:
        return prefix


# single func for easier changes
import subprocess
def compile_binding(name_hash):
    # os.system(f"make -s -C build TARGET={name_hash}.so")
    # os.system(f"make -C build TARGET={name_hash}.so")
    try:
        subprocess.run(['make', '-s', '-C', 'build', f"TARGET={name_hash}.so"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        raise e


def register_class(class_name, namespace, template_args, qualified_name = None):
    if qualified_name is None:
        # use cpp name as qualified name for classes
        qualified_name = get_cpp_name(class_name, namespace, template_args)
    else:
        class_name = qualified_name
        end = class_name.find("<")
        if end != -1:
            class_name=class_name[:end]
        begin = class_name.rfind("::")
        if begin != -1:
            class_name=class_name[begin+2:]

    name_hash = "f_" + hashlib.sha1(qualified_name.encode('utf-8')).hexdigest()
    mod_name = f"build.{name_hash}"
    mod_path = f"build/{name_hash}.so"
    if mod_name not in sys.modules:
        try:
            import_module(mod_path, mod_name)
        except (ImportError, ModuleNotFoundError):
            timer.reset()
            with open(f"build/{name_hash}.cpp", "w") as f:
                f.write(f"#include \"{class_name}.hpp\"\n")

                f.write(f"PYBIND11_MODULE({name_hash}, k){{")
                f.write(f"generate_class<{qualified_name}>(k, \"{name_hash}\", \"{qualified_name}\");}}");

            compile_binding(name_hash)
            global total_build_time
            total_build_time += timer.seconds()

            import_module(mod_path, mod_name)
    return qualified_name


def verify_return_registered(mod):
    try:
        getattr(mod, _DUMMY_RET_FUNC_NAME)()
    except TypeError as e:
        _type_err_patt = re.compile(r"-> (.*)")
        ret_type = re.search(_type_err_patt, str(e)).group(1)
        register_class(None, None, None, qualified_name=ret_type)


_FUNC_NAME = "func"
_DUMMY_RET_FUNC_NAME = "dummy"
def gen_pybind_module(f, binding, name_hash, take_ownership):
    ret_type = f"decltype({_FUNC_NAME}(pybind11::args()))"

    binding.append(
        "template <typename T>"
        f"std::enable_if_t<!std::is_same<T, void>::value, T> {_DUMMY_RET_FUNC_NAME}() {{"
            "return T {};"
        "}"
        "template <typename T>"
        f"std::enable_if_t<std::is_same<T, void>::value, T> {_DUMMY_RET_FUNC_NAME}() {{}}"
    )

    f.writelines(binding)

    return_policy = "automatic" if take_ownership else "automatic_reference"
    f.write(
        f"PYBIND11_MODULE({name_hash}, k){{"
            f"k.def(\"{name_hash}\", &{_FUNC_NAME}, pybind11::return_value_policy::{return_policy});"
            f"k.def(\"{_DUMMY_RET_FUNC_NAME}\", &{_DUMMY_RET_FUNC_NAME}<decltype({_FUNC_NAME}(pybind11::args()))>);"
        "}"
    )


def generate_constructor(class_cpp_name, args, includes):
    generated_ctors.add(class_cpp_name)

    qualified_name = get_qualified_name(class_cpp_name, None, args, None)
    name_hash = "f_" + hashlib.sha1(qualified_name.encode('utf-8')).hexdigest()
    mod_name = f"build.{name_hash}"
    mod_path = f"build/{name_hash}.so"
    if mod_name not in sys.modules:
        try:
            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])
        except (ImportError, ModuleNotFoundError):
            timer.reset()
            with open(f"build/{name_hash}.cpp", "w") as f:
                for include in includes:
                    f.write(f"#include <{include}>\n")

                binding = [
                    f"auto {_FUNC_NAME}(pybind11::args args){{",
                    # "Kokkos::initialize();"
                ]

                # create temp for all args for lifetime 
                gen_arg_casts(binding, args)

                binding.append(f"return new {class_cpp_name} {{")
                vargs = []
                for i, arg in enumerate(args):
                    if isinstance(arg, char_ptr):
                        vargs.append(f"a{i}.c_str()")
                    else:
                        vargs.append(f"a{i}")

                binding.append(",".join(vargs) + "};}")

                gen_pybind_module(f, binding, name_hash, True)

            compile_binding(name_hash)
            global total_build_time
            total_build_time += timer.seconds()

            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])
            
    return (sys.modules[mod_name], name_hash)


def generate_class_func_binding(inst, func_name, args, includes, take_ownership):
    generated_kernels.add(func_name)

    if inst._handle is None:
        raise TypeError("Attempted to call function on type object!")

    qualified_name = get_qualified_name(inst._cpp_name + func_name, None, args, None)
    name_hash = "f_" + hashlib.sha1(qualified_name.encode('utf-8')).hexdigest()
    mod_name = f"build.{name_hash}"
    mod_path = f"build/{name_hash}.so"
    if mod_name not in sys.modules:
        try:
            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])
        except (ImportError, ModuleNotFoundError):
            timer.reset()
            with open(f"build/{name_hash}.cpp", "w") as f:
                for include in includes:
                    f.write(f"#include <{include}>\n")
               
                binding = [
                    f"auto {_FUNC_NAME}(pybind11::args args){{",
                    # "Kokkos::initialize();"
                ]

                # create temp for all args for lifetime 
                gen_arg_casts(binding, (inst,) + args)

                binding.append(f"return a0.{func_name}(")
                vargs = []
                for i, arg in enumerate(args):
                    if isinstance(arg, char_ptr):
                        vargs.append(f"a{i+1}.c_str()")
                    else:
                        vargs.append(f"a{i+1}")

                binding.append(",".join(vargs) + ");}")
                
                gen_pybind_module(f, binding, name_hash, take_ownership)


            compile_binding(name_hash)
            global total_build_time
            total_build_time += timer.seconds()

            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])

    return (sys.modules[mod_name], name_hash)


def generate_operator_binding(inst, args, op_type, includes):
    func_name = op_type.value
    generated_kernels.add(func_name)

    if inst._handle is None:
        raise TypeError("Attempted to call function on type object!")

    qualified_name = get_qualified_name(inst._cpp_name + func_name, None, args, None)
    name_hash = "f_" + hashlib.sha1(qualified_name.encode('utf-8')).hexdigest()
    mod_name = f"build.{name_hash}"
    mod_path = f"build/{name_hash}.so"
    if mod_name not in sys.modules:
        try:
            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])
        except (ImportError, ModuleNotFoundError):
            timer.reset()
            with open(f"build/{name_hash}.cpp", "w") as f:
                for include in includes:
                    f.write(f"#include <{include}>\n")
               
                binding = [
                    f"auto {_FUNC_NAME}(pybind11::args args){{",
                    # "Kokkos::initialize();"
                ]

                # create temp for all args for lifetime 
                gen_arg_casts(binding, (inst,) + args)

                if op_type is Operator.ADD:
                    binding.append(f"return a0 + a1;")
                elif op_type is Operator.SUB:
                    binding.append(f"return a0 - a1;")
                elif op_type is Operator.GET_ITEM:
                    binding.append(f"return a0[a1];")
                elif op_type is Operator.SET_ITEM:
                    binding.append(f"a0[a1] = a2;")
                elif op_type is Operator.CALL:
                    vargs = []
                    for i, arg in enumerate(args):
                        if isinstance(arg, char_ptr):
                            vargs.append(f"a{i+1}.c_str()")
                        else:
                            vargs.append(f"a{i+1}")

                    binding.append(f"return a0({','.join(vargs)});")
                elif op_type is Operator.DEREF:
                    binding.append(f"return *a0;")
                else:
                    raise ValueError(f"Unknown Operator: ", op_type)
                binding.append("}")

                gen_pybind_module(f, binding, name_hash, False)


            compile_binding(name_hash)
            global total_build_time
            total_build_time += timer.seconds()

            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])

    return (sys.modules[mod_name], name_hash)


def generate_func_binding(func_name, namespace, args, includes, template_args, take_ownership):
    generated_kernels.add(func_name)

    qualified_name = get_qualified_name(func_name, namespace, args, template_args)
    name_hash = "f_" + hashlib.sha1(qualified_name.encode('utf-8')).hexdigest()
    mod_name = f"build.{name_hash}"
    mod_path = f"build/{name_hash}.so"
    if mod_name not in sys.modules:
        try:
            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])
        except (ImportError, ModuleNotFoundError) as e:
            timer.reset()
            with open(f"build/{name_hash}.cpp", "w") as f:
                for include in includes:
                    f.write(f"#include <{include}>\n")

                binding = [
                    f"auto {_FUNC_NAME}(pybind11::args args){{",
                    # "Kokkos::initialize();"
                ]

                # create temp for all args for lifetime 
                gen_arg_casts(binding, args)

                # explicit template args
                binding.append(f"return {get_cpp_name(func_name, namespace, template_args)}(")

                vargs = []
                for i, arg in enumerate(args):
                    if isinstance(arg, char_ptr):
                        vargs.append(f"a{i}.c_str()")
                    else:
                        vargs.append(f"a{i}")

                binding.append(",".join(vargs) + ");}")

                gen_pybind_module(f, binding, name_hash, take_ownership)


            compile_binding(name_hash)
            global total_build_time
            total_build_time += timer.seconds()

            import_module(mod_path, mod_name)
            verify_return_registered(sys.modules[mod_name])

    return (sys.modules[mod_name], name_hash)


def cast_return(res):
    if hasattr(res, "_cpp_type"):
        t = res._cpp_type
        end = t.find("<")
        if end != -1:
            t=t[:end]
        begin = t.rfind("::")
        if begin != -1:
            t=t[begin+2:]
        return getattr(sys.modules["kernels"], t)(_handle=res)
    elif type(res).__name__.startswith("KokkosView"):
        # cast views
        params = type(res).__name__.split("_")
        dtype = pk.DataType(getattr(lib, params[1]))
        layout_str = params[3] if params[2] == "const" else params[2]
        if "Layout" not in layout_str:
            layout = Layout.LayoutDefault
        else:
            layout = pk.Layout(getattr(lib, layout_str))
        space = pk.MemorySpace(res.memory_space)
        return pk.View(res.shape, dtype, layout=layout, space=space, array=res)
    else:
        return res

def print_build_info():
    print(f"dynamic_compile_time=[{total_build_time}]")
    print(f"num_ctors=[{len(generated_ctors)}]")
    print(f"num_kernels=[{len(generated_kernels)}]")

import atexit
atexit.register(print_build_info)
