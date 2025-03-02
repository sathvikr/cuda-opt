from clang.cindex import Config, Index, TranslationUnit
import os

# Optionally, set the path to your libclang if it isn’t found automatically.
libclang_path = os.getenv("LIBCLANG_PATH")
if libclang_path:
    # If LIBCLANG_PATH is a directory:
    Config.set_library_path(libclang_path)
    # Alternatively, if you already know the full path to the library file,
    # you can use:
    # Config.set_library_file("/path/to/libclang.so")

def compile_cuda_kernel_to_ast(kernel_code):
    """
    Compiles a CUDA kernel source code into an AST using Clang's Python bindings.

    Parameters:
        kernel_code (str): The CUDA kernel source code.
    
    Returns:
        TranslationUnit: The translation unit representing the AST of the kernel.
    """
    # Prepare Clang arguments to treat the file as CUDA source code.
    args = [
        '-x', 'cuda',               # Specify CUDA language explicitly.
        '--cuda-gpu-arch=sm_70',      # Set the target GPU architecture; adjust as needed.
        '-std=c++14',               # Use a modern C++ standard.
        '-I/usr/local/cuda/include' # Specify the CUDA include directory.
    ]
    
    # Create an index to manage the translation unit.
    index = Index.create()
    
    # Use an unsaved file for in-memory compilation.
    unsaved_files = [('kernel.cu', kernel_code)]
    
    # Parse the source to create the AST.
    translation_unit = index.parse(
        "kernel.cu", 
        args=args, 
        unsaved_files=unsaved_files, 
        options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    )
    
    return translation_unit

def print_ast(cursor, indent=0):
    """
    Recursively prints the AST starting from the given cursor.
    
    Parameters:
        cursor : The AST cursor from which to start.
        indent (int): The current indentation level.
    """
    print(" " * indent + f"{cursor.kind} : {cursor.spelling}")
    for child in cursor.get_children():
        print_ast(child, indent + 2)

# Example usage:
if __name__ == "__main__":
    cuda_kernel_code = """
    __global__ void add(int* a, int* b, int* c) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        c[idx] = a[idx] + b[idx];
    }
    """
    try:
        # Compile the kernel to its AST.
        tu = compile_cuda_kernel_to_ast(cuda_kernel_code)
        # Print the kind of the root AST node (should be a translation unit).
        print("AST Root Kind:", tu.cursor.kind)
        # Optionally, traverse and print the entire AST for inspection.
        print("Full AST:")
        print_ast(tu.cursor)
    except Exception as e:
        print("Error while compiling CUDA kernel to AST:", e)
