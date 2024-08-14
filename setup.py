from setuptools import find_packages, setup

setup(
    name="chatqa",
    version="0.1.0",
    packages=find_packages(include=["./src/*"]),
    python_requires=">=3.11.0",
    install_requires=[
        "overrides",
        "transformers",
        # datasets version fix
        "datasets==2.20.0",
        "evaluate",
        "accelerate",
        "trl",
        "peft",
        "bitsandbytes",
        "wandb",
    ]
)
