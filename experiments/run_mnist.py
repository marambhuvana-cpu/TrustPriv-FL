import os, sys, subprocess
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ds = {'mnist':'MNIST','fashion_mnist':'FashionMNIST','cifar10':'CIFAR10'}['mnist']
subprocess.run([sys.executable, os.path.join(root, 'main.py'), '--config', os.path.join(root, 'config.yaml'), '--dataset', ds, '--mode', 'trustpriv'], check=True)
