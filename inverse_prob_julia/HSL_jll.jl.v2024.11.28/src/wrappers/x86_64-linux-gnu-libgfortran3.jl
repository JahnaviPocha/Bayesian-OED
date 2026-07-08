# Wrapper script for HSL_jll for x86_64-linux-gnu-libgfortran3
export libhsl, libhsl_subset, libhsl_subset_64

using CompilerSupportLibraries_jll
using MPICH_jll
using METIS_jll
using libblastrampoline_jll
JLLWrappers.@generate_wrapper_header("HSL")
JLLWrappers.@declare_library_product(libhsl, "libhsl.so")
JLLWrappers.@declare_library_product(libhsl_subset, "libhsl_subset.so")
JLLWrappers.@declare_library_product(libhsl_subset_64, "libhsl_subset_64.so")
function __init__()
    JLLWrappers.@generate_init_header(CompilerSupportLibraries_jll, MPICH_jll, METIS_jll, libblastrampoline_jll)
    JLLWrappers.@init_library_product(
        libhsl,
        "lib/x86_64-linux-gnu-libgfortran3/libhsl.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libhsl_subset,
        "lib/x86_64-linux-gnu-libgfortran3/libhsl_subset.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libhsl_subset_64,
        "lib/x86_64-linux-gnu-libgfortran3/libhsl_subset_64.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
