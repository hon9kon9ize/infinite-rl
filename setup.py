from setuptools import setup

setup(
    name="infinite_rl",
    version="0.1.13",
    packages=["infinite_rl", "infinite_rl.reward_functions", "infinite_rl.examples"],
    include_package_data=True,
    package_data={
        "infinite_rl.examples": ["*.md"],
        "infinite_rl.reward_functions": ["*.py"],
    },
    install_requires=["wasmtime"],
)
