import copy
import enum
import ctypes
import hashlib
import importlib
import os
import re
import typing
import sys

# benchmark info
import time

class Timer:
    def __init__(self):
        self.start_time: float = time.perf_counter()

    def seconds(self) -> float:
        current_time: float = time.perf_counter()
        return current_time - self.start_time

    def reset(self) -> None:
        self.start_time = time.perf_counter()

total_build_time = 0
timer = Timer()
generated_ctors = set()
generated_kernels = set()

class ptr:
    def __init__(self, val):
        self.val = val 


class char_ptr:
    def __init__(self, val):
        self.val = val 


def get_handle(v):
    if hasattr(v, "_handle"):
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


def get_hash(prefix, namespace, args, template_args):
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

    qualified_name = f"{prefix}{template_str}({','.join(cpp_args)})"

    return "f_" + hashlib.sha1(qualified_name.encode('utf-8')).hexdigest()


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


def call_constructor(cls_inst, args, includes):
    if cls_inst._handle:
        if hasattr(cls_inst, '__cpp_call__'):
            return cls_inst.__cpp_call__(*args)
        raise RuntimeError("Error: calling constructor on instance is forbidden!")

    # check binding and generate
    class_cpp_name = cls_inst._cpp_name
    generated_ctors.add(class_cpp_name)

    name_hash = get_hash(class_cpp_name, None, args, None)
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
            
    # constructor invocation
    args = [get_handle(arg) for arg in args]
    inst = copy.copy(cls_inst)
    inst._handle = getattr(sys.modules[mod_name], name_hash)(*args)
    return inst


def call_class_func(inst, func_name, args, includes, take_ownership):
    # check binding and generate
    generated_kernels.add(func_name)

    if inst._handle is None:
        raise TypeError("Attempted to call function on type object!")

    name_hash = get_hash(inst._cpp_name + func_name, None, args, None)
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

    # func invocation
    args = [get_handle(arg) for arg in args]
    res = getattr(sys.modules[mod_name], name_hash)(inst._handle, *args)
    return cast_return(res)


def generate_operator_binding(inst, args, op_type, includes):
    func_name = op_type.value
    generated_kernels.add(func_name)

    if inst._handle is None:
        raise TypeError("Attempted to call function on type object!")

    name_hash = get_hash(inst._cpp_name + func_name, None, args, None)
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


def call_func(func_name, namespace, args, includes, template_args, take_ownership):
    # check binding and generate
    generated_kernels.add(func_name)

    name_hash = get_hash(func_name, namespace, args, template_args)
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

    # func invocation
    args = [get_handle(arg) for arg in args]
    res = getattr(sys.modules[mod_name], name_hash)(*args)
    return cast_return(res)


# register manually written bindings
# assumes all binding names are of the format <class>_<t1>_<t2>_...
_custom_types = {}
def register_manual_wrapper(wrapper_cls, cls_name):
    assert hasattr(wrapper_cls, "_handle")
    assert hasattr(wrapper_cls, "_cpp_name")
    assert hasattr(wrapper_cls, "_from_handle")
    _custom_types[cls_name] = wrapper_cls._from_handle

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

    #TODO: find more flexible mechanism
    t = type(res).__name__.split("_")[0]
    if t in _custom_types:
        return _custom_types[t](res)
    else:
        return res

def print_build_info():
    print(f"dynamic_compile_time=[{total_build_time}]")
    print(f"num_ctors=[{len(generated_ctors)}]")
    print(f"num_kernels=[{len(generated_kernels)}]")

import atexit
atexit.register(print_build_info)
