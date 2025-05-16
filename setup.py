from setuptools import setup, find_packages

setup(
    name='omni_ieeg',
    version='0.1.0',
    description='A package for downloading, processing, and running Omni-iEEG data',
    author='Chenda Duan',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pyyaml',
        'openpyxl',
        'HFODetector',
        "scipy",
        "pandas",
        "tqdm",
        "numpy",
        "torch",
        "scikit-image",
        "openpyxl",
        "HFODetector",
        "p_tqdm",
        "torchvision",
        "mne",
        "edfio",
        "transformers",
        "pyarrow",
        "einops"
    ],
)