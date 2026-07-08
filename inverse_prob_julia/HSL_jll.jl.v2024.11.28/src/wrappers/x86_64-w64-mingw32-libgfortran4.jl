# Wrapper script for HSL_jll for x86_64-w64-mingw32-libgfortran4
export libhsl, libhsl_subset, libhsl_subset_64

using CompilerSupportLibraries_jll
using MicrosoftMPI_jll
using METIS_jll
using libblastrampoline_jll
JLLWrappers.@generate_wrapper_header("HSL")
JLLWrappers.@declare_library_product(libhsl, "libhsl.dll")
JLLWrappers.@declare_library_product(libhsl_subset, "libhsl_subset.dll")
JLLWrappers.@declare_library_product(libhsl_subset_64, "libhsl_subset_64.dll")
function __init__()
    JLLWrappers.@generate_init_header(CompilerSupportLibraries_jll, MicrosoftMPI_jll, METIS_jll, libblastrampoline_jll)
    JLLWrappers.@init_library_product(
        libhsl,
        "bin\\x86_64-w64-mingw32-libgfortran4\\libhsl.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libhsl_subset,
        "bin\\x86_64-w64-mingw32-libgfortran4\\libhsl_subset.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libhsl_subset_64,
        "bin\\x86_64-w64-mingw32-libgfortran4\\libhsl_subset_64.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
