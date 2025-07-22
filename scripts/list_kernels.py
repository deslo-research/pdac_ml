from jupyter_client.kernelspec import KernelSpecManager

def list_kernels():
    ksm = KernelSpecManager()
    specs = ksm.get_all_specs()
    
    print("Available kernels:")
    for name, spec in specs.items():
        print(f"  {name}: {spec['spec']['display_name']}")
        print(f"     {spec['resource_dir']}")

if __name__ == "__main__":
    list_kernels()