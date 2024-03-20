from setuptools import setup, find_packages

setup(
    name='s3datasets',
    python_requires='>=3.6',
    description = "System to ease incremental training of a Huggingface transformer model from a large S3-based dataset",
    long_description="This Python module provides tools to enhance Huggingface Trainer to make it feasible to train transformers on very large remote datasets that are impractical to just download en-masse for every training session or to complete all of the training on a single server instance.  It makes it practical to use cloud providers like vast.ai or salad.ai as a fine-tuning platform for large language models.",
    version='0.0.1a1',
    package_dir={'': 'src'},  # This tells setuptools where to find packages
    packages=find_packages(where='src'),
    install_requires=[
        's3datasets',
        'dataclasses',
        'typing; python_version<"3.5"',
        'peft'
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]

)


