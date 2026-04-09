from setuptools import setup, find_packages

setup(
    name="mars_mission",
    version="0.1.0",
    description="Earth-to-Mars multi-objective trajectory optimization — 3M Aerospace Solutions",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "astropy",
        "pymoo",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "dev": ["pytest", "jupyter", "ipykernel"],
    },
)