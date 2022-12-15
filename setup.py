from setuptools import setup

__version__ = (0, 0, 0)

setup(
    name="far_range_upsampling",
    description="LiDAR point cloud enhancement in the far range from the ego vehicle for autonomous driving",
    version=".".join(str(d) for d in __version__),
    author="Sangwon Lim",
    author_email="sangwon3@ualberta.ca",
    packages=["far_range_upsampling"],
    include_package_data=True,
    scripts="""
        ./scripts/train
        ./scripts/test_models
        ./scripts/predict
    """.split(),
)
