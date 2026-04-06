from setuptools import setup, find_packages

setup(
    name="mars_transfer",
    version="0.1.0",
    description="Multi-objective Earth-to-Mars trajectory optimization for 3M Aerospace Solutions",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "astropy",   # ephemeris and time
        "pymoo",     # NSGA-II multi-objective optimizer
        "numpy",
        "scipy",     # brentq for Lambert solver
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "dev": ["pytest", "jupyter", "ipykernel"],
    },
)
