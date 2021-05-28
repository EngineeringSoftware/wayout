"""
static.py

handles generation of wrapper classes
"""

# TODO: move function body into python library rather than generated?

from collections import deque
from pathlib import Path
import os
import re
import sys
import subprocess

import clang.cindex as cindex

namespace = None
def traverse_ast(output, path, node, indent, functions, enums, build_dir, headers):
    global namespace
    temp_namespace = namespace 

    # ignore c++ stdlib
    if node.location.file and node.location.file.name != path:
        return
    # if node.spelling in ['std', "__gnu_cxx", "__gnu_debug"] or node.kind == cindex.CursorKind.UNEXPOSED_DECL:
    #     return

    # print("--" * indent, node.spelling, node.kind)
    if node.kind == cindex.CursorKind.NAMESPACE:
        temp_namespace = namespace
        if namespace != None:
            namespace = namespace + "::" + node.spelling
        else:
            namespace = node.spelling

    if node.kind == cindex.CursorKind.CLASS_TEMPLATE:
        # only parse class definitions
        if not node.is_definition():
            return
        generate_class_header(node, build_dir, headers)
        generate_class(output, node)
        output.append("")
        return
    elif node.kind == cindex.CursorKind.FUNCTION_TEMPLATE or node.kind == cindex.CursorKind.FUNCTION_DECL:
        name = node.spelling
        # ignore non-member operator overloading
        if "operator" in node.spelling:
            return 

        # disallow illegal name (templated structs)
        if "<" in name:
            return

        if name in functions:
            functions[name].append(node)
        else:
            functions[name] = [namespace, node]
        return
    elif node.kind == cindex.CursorKind.ENUM_DECL:
        enums.append((namespace, node))
        return

    # Recurse for children of this node
    for c in node.get_children():
        traverse_ast(output, path, c, indent+1, functions, enums, build_dir, headers)

    namespace = temp_namespace 


# get params of a node and convert to templated type
def get_params(node, func_node):
    params = []
    for param in func_node.get_children():
        if param.kind == cindex.CursorKind.PARM_DECL:
            # if param.spelling == "":
            #     param_name = f"_param{len(params)}" 
            # else:
            #     param_name = f"{param.spelling}"

            p_type = param.type 
            # handle array first
            if p_type.kind == cindex.TypeKind.INCOMPLETEARRAY:
                p_type = p_type.get_array_element_type()

            # pointer
            if p_type.kind == cindex.TypeKind.LVALUEREFERENCE or \
                p_type.kind == cindex.TypeKind.POINTER:
                param_type = p_type.get_pointee().spelling
            else:
                param_type = p_type.spelling

            # const
            if p_type.is_const_qualified():
                param_type = param_type.replace("const ", "")

            # class type
            if f"{node.spelling}::" in param_type:
                t = param_type.split("::")[-1]
                param_type = param.type.spelling.replace(param_type, f"typename T_::{t}")
            else:
                param_type = param.type.spelling

            # template arg
            # if param_type in template_args:
            #     param_type = param.type.spelling.replace(param_type, f"typename T_::{param_type}")

            params.append(param_type)
    return params


# generate guarded invocation to potentially register return types
def gen_guarded_invocation(output, tab_lvl, invocation_str):
    lines = [
        "try:",
        f"\tres = {invocation_str}",
        "except TypeError as e:",
        "\tret_type = re.search(_type_err_patt, str(e)).group(1)",
        "\tregister_class(None, None, None, qualified_name=ret_type)",
        f"\tres = {invocation_str}",
    ]

    output.extend(["\t"*tab_lvl + line for line in lines])


def generate_class_header(node, build_dir, headers):
    header = []
    header.extend([f"#include <{f}>" for f in headers])

    template_args = ["class T_"]
    # provide access to template args
    # for c in node.get_children(): 
    #     if c.kind == cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
    #         template_args.append("class " + c.spelling)

    header.extend([
        f"template <{','.join(template_args)}>",
        "void generate_class(pybind11::module &_mod, const char *name, const char *cpp_type) {",
        "   pybind11::class_<T_> _class(_mod, name);",
        "   _class.def_property_readonly_static(\"_cpp_type\", [cpp_type](const pybind11::object&) { return cpp_type; });"
    ])
    
    for c in node.get_children(): 
        # public class variable
        if c.kind == cindex.CursorKind.FIELD_DECL and \
                c.access_specifier == cindex.AccessSpecifier.PUBLIC:
            if c.spelling == "__device__":
                continue

            def_func = "readwrite" 
            if c.type.is_const_qualified():
                def_func = "readonly"
            header.append(f"  _class.def_{def_func}(\"{c.spelling}\", &T_::{c.spelling});")
    header.append("}");

    with open(f"{build_dir}/{node.spelling}.hpp", "w") as f:
        f.write("\n".join(header))


# get only class name from templated name (potentitally typedef)
def get_class_name(name):
    
    typedef = name.startswith("typename")
    name = name.replace("typename ", '').replace(" ",'')

    # check for implicit typedef
    if not typedef:
        has_template = "<" in name
        # check last word is pure name (contains no <>)
        is_pure_name = ">" not in name.split("::")[-1] 
        typedef = has_template and is_pure_name

    # remove templates
    n = 1
    while n:
        name, n = re.subn(r'\([^()]*\)', '', name)
    n = 1
    while n:
        name, n = re.subn(r'<[^<>]*>', '', name)

    parts = name.split("::")
    if len(parts) == 1:
        return parts[0]

    if typedef:
        return parts[-2]+"."+parts[-1]
    else:
        return parts[-1]


def generate_class(output, node):
    """
    template_args = []
    # define template types
    targ_count = 0
    for c in node.get_children(): 
        if c.kind == cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
            targ = c.spelling if c.spelling else f"t{targ_count}"
            output.append(f"{targ} = TypeVar('{targ}')")
            template_args.append(targ)
            targ_count += 1

    if len(template_args) > 0:
        output.append(f"class {node.spelling}(Generic[{','.join(template_args)}]):")
    else:
        output.append(f"class {node.spelling}:")
    """
    # get parents
    parents = []
    for c in node.get_children(): 
        if c.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
            if node.spelling == "normal_iterator":
                print("normal iterator parent", c.spelling)
            parents.append(get_class_name(c.spelling))

    # FIXME temp hardcode for normal iterator parent (doesn't show up in ast)
    if node.spelling == "normal_iterator":
        parents = ["iterator_adaptor"]
    elif node.spelling == "counting_iterator":
        parents = ["counting_iterator_base.type"]

    # FIXME clang bug?
    # ["generate_functor", "destroy_functor", "category_to_traversal", ...]:

    if len(parents) > 1:
        raise NotImplementedError("Multiple Inheritance currently not supported!")
        # parents = parents[:1]
    if len(parents) != 0:
        output.extend([
            "try:",
                f"\t{parents[0]}",
            "except (AttributeError, NameError):",
                f"\t{parents[0]} = object",
        ])

    # class def
    output.append(f"class {node.spelling}({','.join(parents)}):")
    if node.brief_comment:
        output.append(f"\t\"\"\"{node.brief_comment}\"\"\"")

    # namespace
    output.append(f"\t_namespace = \"{namespace}\"")

    # class body
    # register class in init 
    output.extend(["\tdef __init__(self, *template_args, _handle=None):",
        "\t\tself._handle = _handle",
        f"\t\tself._cpp_name = _handle._cpp_type if _handle else register_class(\"{node.spelling}\", self._namespace, template_args)"
        ""
    ])

    # call constructor
    output.extend(["\tdef __call__(self, *args):",
        "\t\tif self._handle:",
            f"\t\t\tif hasattr(self, '__cpp_call__'):",
                f"\t\t\t\treturn self.__cpp_call__(*args)",
            "\t\t\traise RuntimeError(\"Error: calling constructor on instance is forbidden!\")",
        f"\t\tmod, name = generate_constructor(self._cpp_name, args, _includes)",
        f"\t\targs = [get_handle(arg) for arg in args]",
        f"\t\tinst = _copy.copy(self)",
        f"\t\tinst._handle = getattr(mod, name)(*args)",
        "\t\treturn inst",
        ""
    ])

    functions = []

    for c in node.get_children(): 
        # public methods
        child_name = c.spelling
        if (c.kind == cindex.CursorKind.CXX_METHOD or c.kind == cindex.CursorKind.FUNCTION_TEMPLATE) and \
                c.access_specifier == cindex.AccessSpecifier.PUBLIC:
            if c.kind == cindex.CursorKind.FUNCTION_TEMPLATE:
                idx = child_name.find("<")
                name = child_name[:idx] if idx > 0 else child_name
                # ignore constructors
                if name == node.spelling:
                    continue
                     
            # skip overloaded functions, handled by dynamic generation
            if child_name in functions:
                continue

            functions.append(child_name)

            # implement operator overloading
            if "operator" in child_name:
                supported_operators = ["operator+", "operator-", "operator[]", "operator()", "operator*"]
                if child_name in supported_operators:
                    if c.raw_comment:
                        comment = '\t'.join(c.raw_comment.splitlines(True))
                        output.append(f"\t\"\"\"{comment}\"\"\"")
                else:
                    # if "()" not in child_name:
                    #     print(child_name)
                    continue

                # addition operator
                if child_name == "operator+":
                    output.append(f"\tdef __add__(self, other):")
                    output.extend([
                        f"\t\tmod, name = generate_operator_binding(self, (other,), Operator.ADD, _includes)",
                        "\t\tres = getattr(mod, name)(self._handle, get_handle(other))",
                        "\t\treturn cast_return(res)"
                    ])
                # subtraction operator
                elif child_name == "operator-":
                    output.append(f"\tdef __sub__(self, other):")
                    output.extend([
                        f"\t\tmod, name = generate_operator_binding(self, (other,), Operator.SUB, _includes)",
                        "\t\tres = getattr(mod, name)(self._handle, get_handle(other))",
                        "\t\treturn cast_return(res)"
                    ])
                # indexing operator
                elif child_name == "operator[]":
                    output.append(f"\tdef __getitem__(self, key):")
                    output.extend([
                        f"\t\tmod, name = generate_operator_binding(self, (key,), Operator.GET_ITEM, _includes)",
                        "\t\tres = getattr(mod, name)(self._handle, key)",
                        "\t\treturn cast_return(res)"
                    ])

                    output.append(f"\tdef __setitem__(self, key, val):")
                    output.extend([
                        f"\t\tmod, name = generate_operator_binding(self, (key, val), Operator.SET_ITEM, _includes)",
                        "\t\tres = getattr(mod, name)(self._handle, key, val)",
                        "\t\treturn cast_return(res)"
                    ])
                # call operator 
                elif child_name == "operator()":
                    output.append(f"\tdef __cpp_call__(self, *args):")
                    output.extend([
                        f"\t\tmod, name = generate_operator_binding(self, args, Operator.CALL, _includes)",
                        f"\t\targs = [get_handle(arg) for arg in args]",
                        "\t\tres = getattr(mod, name)(self._handle, *args)",
                        "\t\treturn cast_return(res)"
                    ])
                # deref operator 
                elif child_name == "operator*":
                    output.append(f"\tdef __deref__(self):")
                    output.extend([
                        f"\t\tmod, name = generate_operator_binding(self, (), Operator.DEREF, _includes)",
                        "\t\tres = getattr(mod, name)(self._handle)",
                        "\t\treturn cast_return(res)"
                    ])

                continue


            params = ["self"]
            for param in c.get_children():
                if param.kind == cindex.CursorKind.PARM_DECL:
                    if param.spelling == "":
                        params.append(f"_param{len(params)}")
                    else:
                        params.append(f"{param.spelling}")

            output.append(f"\tdef {child_name}(self, *args, take_ownership=False):")
            if c.raw_comment:
                comment = '\t\t'.join(c.raw_comment.splitlines(True))
                output.append(f"\t\t\"\"\"{comment}\"\"\"")

            output.extend([
                f"\t\tmod, name = generate_class_func_binding(self, \"{child_name}\", args, _includes, take_ownership)",
                "\t\targs = [get_handle(arg) for arg in args]",
            ])
            
            # gen_guarded_invocation(2, "getattr(mod, name)(self._handle, *args)")
            output.append("\t\tres = getattr(mod, name)(self._handle, *args)")
            output.append("\t\treturn cast_return(res)")

        # public class variable
        elif c.kind == cindex.CursorKind.FIELD_DECL and \
                c.access_specifier == cindex.AccessSpecifier.PUBLIC:
            if child_name == "__device__":
                continue
            output.append(f"\t@property")

            output.append(f"\tdef {child_name}(self):")
            if c.raw_comment:
                comment = '\t\t'.join(c.raw_comment.splitlines(True))
                output.append(f"\t\t\"\"\"{comment}\"\"\"")

            gen_guarded_invocation(output, 2, f"self._handle.{child_name}")
            output.append("\t\treturn cast_return(res)")

        # typedefs
        elif c.kind == cindex.CursorKind.TYPEDEF_DECL:
            typedef = get_class_name(c.underlying_typedef_type.spelling)
            # remove references
            typedef = typedef.replace("&", "").replace("*","")
            output.extend([
                "\ttry:",
                    f"\t\t{child_name} = {typedef}",
                "\texcept (AttributeError, NameError):",
                    "\t\tpass",
            ])

    # FIXME temp hardcode for discard iterator typedef (doens't show up in ast)
    if node.spelling == "discard_iterator_base" or node.spelling == "counting_iterator_base":
        output.append("\ttype = iterator_adaptor")


def generate_functions(output, functions):
    for name, nodes in functions.items():
        namespace = f"\"{nodes[0]}\"" if nodes[0] else None
        for node in nodes[1:]:
            params = []
            # define template types
            for c in node.get_children(): 
                if c.kind == cindex.CursorKind.PARM_DECL:
                    if c.spelling == "":
                        params.append(f"_param{len(params)}")
                    else:
                        params.append(f"{c.spelling}")
            output.append(f"# {name}({', '.join(params)})")

            if node.raw_comment:
                output.append(f"\"\"\"{node.raw_comment}\"\"\"")

        output.extend([
            f"def {name}(*args, template_args=None, take_ownership=False, namespace={namespace}):",
            f"\tmod, name = generate_func_binding(\"{name}\", namespace, args, _includes, template_args, take_ownership)",
            "\targs = [get_handle(arg) for arg in args]",
        ])

        # gen_guarded_invocation(1, "getattr(mod, name)(*args)")
        output.append("\tres = getattr(mod, name)(*args)")
        output.append("\treturn cast_return(res)")
        output.append("")


def generate_enums(build_dir, nodes, headers):
    enum_lines = [f"#include <{f}>" for f in headers]
    enum_lines.append("PYBIND11_MODULE(_kernel_enums, k) {")
    for namespace, node in nodes:
        if not node.spelling:
            continue
        enum_lines.append(f"pybind11::enum_<{namespace}::{node.spelling}>(k, \"{node.spelling}\", \"{namespace}\")")
        for field in node.get_children():
            enum_lines.append(f".value(\"{field.spelling}\", {namespace}::{node.spelling}::{field.spelling})")
        enum_lines.append(".export_values();")

    enum_lines.append("}")
    with open(f"{build_dir}/_kernel_enums.cpp", "w") as f:
        f.write("\n".join(enum_lines))

    # os.system(f"make -C {build_dir} TARGET=_kernel_enums.so > /dev/null 2>&1")
    # os.system(f"make -C {build_dir} TARGET=_kernel_enums.so > /dev/null")
    try:
        subprocess.run(['make', '-s', '-C', build_dir, f"TARGET=_kernel_enums.so"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        raise e


import enum
class Target(enum.Enum):
    kokkos_omp = "Makefile.kokkos_omp"
    kokkos_cuda = "Makefile.kokkos_cuda"
    thrust_omp = "Makefile.thrust_omp"
    thrust_cuda = "Makefile.thrust_cuda"

import pykokkos as pk
def generate_wrapper(output_dir, paths, flags=[], target=Target.kokkos_omp):
    output = []
    #FIXME need general way of getting header file name
    if "thrust" in output_dir:
        #FIXME temp header to init vectors for benchmarks
        init_vec = str(Path("init_vec.hpp").resolve())
        headers = ["pybind11/pybind11.h", init_vec] + [path[path.rfind("thrust/"):] for path in paths]
        paths.append(init_vec)
    else:
        headers = ["pybind11/pybind11.h", ] + [os.path.basename(path) for path in paths]
    include_str = '[' + ",".join([f"\"{f}\"" for f in headers]) + ']'

    # cindex.Config.set_library_file(LIB_PATH)
    index = cindex.Index.create()
    output.extend([
        "import copy as _copy",
        "import re",
        "import sys",

        "import wayout",
        "from wayout.dynamic import *",
        "from build._kernel_enums import *",
        f"_includes = {include_str}",
        "",
        "_type_err_patt = re.compile(r\"-> (.*)\")",
        ""
    ])

    build_dir = output_dir + "/build/"
    makefile_path = Path(__file__).resolve().parent / target.value

    os.system(f"mkdir -p {build_dir}")
    os.system(f"cp {makefile_path} {build_dir}/Makefile")

    enums = []
    parsed_paths = set(paths)
    queue = deque(paths)
    while len(queue) > 0:
        path = str(Path(queue.popleft()).resolve())
        tu = index.parse(path, args=flags)
        """
        # recursively add includes
        for sub in tu.get_includes():
            inc_path = sub.include.name
            if not inc_path.startswith("/usr") and inc_path not in parsed_paths:
                parsed_paths.add(inc_path)
                queue.appendleft(inc_path)
        """

        output.append('# Translation unit:'+tu.spelling)
        functions = {}
        traverse_ast(output, path, tu.cursor, 0, functions, enums, build_dir, headers)
        generate_functions(output, functions)

    generate_enums(build_dir, enums, headers)

    with open(output_dir + "/kernels.py", "w") as f:
        f.write("\n".join(output))
    # print("\n".join(output))

