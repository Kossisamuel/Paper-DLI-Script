from importlib.metadata import PackageNotFoundError, version
for name in ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'scipy', 'seaborn', 'matplotlib', 'ucimlrepo', 'nbformat', 'nbclient', 'jupyterlab', 'ipython', 'ipykernel', 'ruff']:
    try:
        print(f'{name}=={version(name)}', flush=True)
    except PackageNotFoundError:
        print(f'{name}: missing', flush=True)
