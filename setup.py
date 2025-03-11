from setuptools import setup, find_packages

setup(
    name="vertebra_feature_extractor",
    version="0.1.0",
    description="A feature extractor for vertebra segmentation analysis.",
    author="Blanca Inigo Romillo",
    author_email="binigo1@jh.edu",
    packages=find_packages(where="vertebra_feature_extractor/src"),
    package_dir={"": "vertebra_feature_extractor/src"},
    install_requires=[
        "omegaconf>=2.0.0",
        "rich>=10.0.0",
        # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            "vertebra-feature-extractor=main:main"
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
