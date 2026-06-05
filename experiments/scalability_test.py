import os, sys, subprocess
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for clients in [20, 40, 60, 80, 100]:
    subprocess.run([sys.executable, os.path.join(root, 'main.py'), '--config', os.path.join(root, 'config.yaml'), '--clients', str(clients), '--rounds', '20', '--mode', 'trustpriv'], check=True)
