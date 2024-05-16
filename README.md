[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Deploying Google PaliGemma on Amazon SageMaker

This tutorial will guide you through the process of deploying the Google PaliGemma vision model on Amazon SageMaker as a real-time inference endpoint using a Jupyter Notebook.

## What is PaliGemma?

PaliGemma is a large-scale vision-language model developed by Google. It can perform various tasks, including image segmentation, object detection, and image captioning, by combining vision and language representations.

## Prerequisites

Before proceeding with this tutorial, ensure that you have the following prerequisites:

- An AWS account with appropriate permissions to create and manage SageMaker resources.
- A Hugging Face account and access token (https://huggingface.co/settings/tokens).

## Step-by-Step Guide

1. **Set up the environment**: The notebook starts by installing the required Python packages, such as `huggingface_hub` and `sagemaker`.

```python
!pip install --upgrade huggingface_hub
!pip install --upgrade sagemaker
```

2. **Authenticate with AWS**: The notebook authenticates with your AWS account by obtaining the SageMaker execution role and default bucket. This step is necessary for SageMaker to have the required permissions to create resources.

```python
import sagemaker
import boto3
sess = sagemaker.Session()
# ... (code omitted for brevity)
```

3. **Prepare the model directory**: The notebook creates a directory structure to store the PaliGemma model files. It also writes the necessary code files (`inference.py` and `requirements.txt`) to this directory.

```python
!mkdir code
```

```python
%%writefile code/requirements.txt
accelerate
bitsandbytes
git+https://github.com/huggingface/transformers.git
Pillow
```

```python
%%writefile code/inference.py
# ... (inference script code omitted for brevity)
```

4. **Download the model snapshot**: The notebook downloads the PaliGemma model snapshot from the Hugging Face repository using your access token. This step requires you to accept the terms and conditions for the PaliGemma model on the Hugging Face website.

```python
from distutils.dir_util import copy_tree
from pathlib import Path
from huggingface_hub import snapshot_download
import random
HF_MODEL_ID="google/PaliGemma-3b-mix-224"
HF_TOKEN="YOUR_HF_TOKEN"

# download snapshot
snapshot_dir = snapshot_download(repo_id=HF_MODEL_ID, use_auth_token=HF_TOKEN)
# ... (code omitted for brevity)
```

5. **Create a compressed model archive**: The notebook compresses the model directory into a `.tar.gz` file, which will be uploaded to Amazon S3 for deployment.

```python
import tarfile
import os

def compress(tar_dir=None, output_file="model.tar.gz"):
    # ... (code omitted for brevity)

compress(str(model_tar))
```

6. **Upload the model to S3**: The compressed model archive is uploaded to an S3 bucket associated with your SageMaker session.

```python
from sagemaker.s3 import S3Uploader

s3_model_uri = S3Uploader.upload(local_path="model.tar.gz", desired_s3_uri=f"s3://{sess.default_bucket()}/paligemma")
```

7. **Deploy the model to SageMaker**: The notebook creates a `HuggingFaceModel` instance and deploys it to a SageMaker endpoint. This step involves specifying the model data location (S3 URI), execution role, and various configuration parameters like instance type and transformers version.

```python
from sagemaker.huggingface.model import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    model_data=s3_model_uri,
    role=role,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version='py310',
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name=f"PaliGemma-{timestamp}"
)
```

8. **Test the deployed model**: The notebook demonstrates how to send a request to the deployed SageMaker endpoint for image detection. It encodes a test image as base64, includes a textual prompt, and sends it to the endpoint for inference. The response from the endpoint is then printed.

```python
import base64
from PIL import Image

# ... (code omitted for brevity)

payload = {
    "prompt": "detect dog",
    "image": encoded_input_image
}

query_response = predictor.predict(data=payload)
```

By following this tutorial and referencing the provided code snippets, you will learn how to deploy the PaliGemma vision model on Amazon SageMaker and test it with a sample image segmentation task. Remember to replace placeholders like `YOUR_HF_TOKEN` with your actual Hugging Face access token.

9. **Delete the deployed model**: running the last command you can delete the endpoint to avoid not desidered costs:

```python
# delete endpoint
predictor.delete_endpoint()
```


## General Disclaimer

This notebook is intended for demonstration and educational purposes only. It is not designed for production use without further modifications and hardening. Before deploying this endpoint to a production environment, it is crucial to conduct thorough testing, security assessments, and optimizations based on your specific requirements and best practices.

## Contributing

Contributions are welcome! Please follow the usual Git workflow:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes
4. Push to the branch
5. Create a new pull request

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

- [License](LICENSE) of the project.
- [Code of Conduct](CODE_OF_CONDUCT.md) of the project.
- [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Legal Disclaimer

You should consider doing your own independent assessment before using the content in this repository for production purposes. This may include (amongst other things) testing, securing, and optimizing the content provided in this repository, based on your specific quality control practices and standards.
