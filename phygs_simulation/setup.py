from setuptools import find_packages, setup


setup(
    name="interactive-search",
    version="0.1.0",
    description="Interactive Search benchmark helpers and Spot skills for Isaac Lab.",
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
