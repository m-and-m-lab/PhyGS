from setuptools import find_packages, setup


setup(
    name="phygs-simulation",
    version="0.1.0",
    description="PhyGS simulation: benchmark helpers and Spot skills for Isaac Lab.",
    package_dir={"": "scripts"},
    packages=find_packages(
        where="scripts",
        include=[
            "helpers",
            "helpers.*",
            "skills",
            "skills.*",
        ],
    ),
    package_data={
        "helpers": ["config/*.yaml"],
        "skills": [
            "locomotion/*.yaml",
            "manipulation/config/*.yaml",
        ],
    },
    include_package_data=True,
    python_requires=">=3.11",
)
