import os, sys, subprocess
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for mode in ['fedavg', 'static', 'heuristic', 'personalized', 'trustpriv']:
    subprocess.run([sys.executable, os.path.join(root, 'main.py'), '--config', os.path.join(root, 'config.yaml'), '--mode', mode, '--rounds', '20'], check=True)
