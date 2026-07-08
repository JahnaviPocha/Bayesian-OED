# Wrapper script for HSL_jll for x86_64-apple-darwin-libgfortran4
export libhsl, libhsl_subset, libhsl_subset_64

using CompilerSupportLibraries_jll
using MPICH_jll
using METIS_jll
using libblastrampoline_jll
JLLWrappers.@generate_wrapper_header("HSL")
JLLWrappers.@declare_library_product(libhsl, "@rpath/libhsl.dylib")
JLLWrappers.@declare_library_product(libhsl_subset, "@rpath/libhsl_subset.dylib")
JLLWrappers.@declare_library_product(libhsl_subset_64, "@rpath/libhsl_subset_64.dylib")
function __init__()
    JLLWrappers.@generate_init_header(CompilerSupportLibraries_jll, MPICH_jll, METIS_jll, libblastrampoline_jll)
    JLLWrappers.@init_library_product(
        libhsl,
        "lib/x86_64-apple-darwin-libgfortran4/libhsl.dylib",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libhsl_subset,
        "lib/x86_64-apple-darwin-libgfortran4/libhsl_subset.dylib",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libhsl_subset_64,
        "lib/x86_64-apple-darwin-libgfortran4/libhsl_subset_64.dylib",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
