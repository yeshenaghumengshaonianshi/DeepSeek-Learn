from setuptools import setup, find_packages

setup(
    name="vllm",
    version="0.2.0",
    description="高效LLM推理和服务引擎",
    author="vLLM开发团队",
    author_email="example@example.com",
    url="https://github.com/vllm-project/vllm",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.33.0",
        "accelerate",
        "sentencepiece",
        "fastapi",
        "uvicorn",
        "pydantic",
        "ray>=2.5.1",
        "numpy>=1.24.0",
        "xformers>=0.0.22",
        "einops",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "mypy",
            "ruff",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 