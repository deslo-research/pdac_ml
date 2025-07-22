import sys
import os
import json
from jupyter_client.kernelspec import KernelSpecManager, KernelSpec

def get_venv_path():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.prefix
    raise Exception("Not running inside virtual env")

def main():
    try:
        kernel_name = "vision-feature-interpreter"
        display_name = "Vision Feature Kernel"
        python_executable = sys.executable

        kernel_spec = KernelSpec()
        kernel_spec.display_name = display_name
        kernel_spec.language = 'python'
        kernel_spec.argv = [python_executable, "-m", "ipykernel_launcher", "-f" "{connection_file}"]

        venv_path = get_venv_path()

        kernel_spec_dir = os.path.join(venv_path, "share", "jupyter", "kernels", kernel_name)
        os.makedirs(kernel_spec_dir, exist_ok=True)

        with open(os.path.join(kernel_spec_dir, "kernel.json"), 'w') as f:
            json.dump(kernel_spec.to_json(), f, indent=2)

        print(f"Kernel spec dir: {kernel_spec_dir}")


        ksm = KernelSpecManager()
        dest = ksm.install_kernel_spec(
            kernel_spec_dir,
            kernel_name=kernel_name,
            user=True
        )
        print(dest)

        print(f"Installed {kernel_name}")

    except Exception as e:
        print(f"Failed to execute subprocess [ERROR]: {str(e)}")
        sys.exit(-1)

if __name__ == "__main__":
    main()
    sys.exit(0)
