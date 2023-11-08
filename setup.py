from setuptools import setup, find_packages

setup(
    name="llm_recipes",
    version="0.0.1",
    packages=find_packages(),
    license="MIT",
    description="A toolbox of llm stuff",
    author="Thomas Capelle",
    author_email="tcapelle@pm.me",
    url="https://github.com/tcapelle/llm_recipes",
    long_description_content_type="text/markdown",
    keywords=[
        "artificial intelligence",
        "generative models",
        "natural language processing",
        "openai",
    ],
    install_requires=[
        "wandb",
        "transformers",
        "datasets",
        "accelerate",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)