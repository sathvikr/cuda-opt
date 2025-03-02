# CUDA Optimizer

## Project Description

Cuda Optimizer is a tool that uses AI to automatically improve the performance of CUDA GPU kernels. Given an input CUDA C/C++ kernel, the optimizer searches relevant literature (ArXiv research papers) and leverages large language models (LLMs) to generate optimized versions of the kernel code ([cuda-opt/driver.py at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/driver.py#:~:text=,papers%20and%20LLMs)). The goal is to help developers obtain faster GPU code without manual fine-tuning by providing **AI-generated CUDA kernels** as output ([GitHub - sathvikr/cuda-opt: AI-generated CUDA kernels.](https://github.com/sathvikr/cuda-opt#:~:text=About)). This project can suggest multiple optimized kernel variants for a given input, integrating state-of-the-art optimization strategies discovered from research and forums into the generated code.

## Installation Instructions

1. **Clone the Repository:** Clone the `cuda-opt` project to your local machine and enter the directory.  
   ```bash
   git clone https://github.com/sathvikr/cuda-opt.git  
   cd cuda-opt
   ```  

2. **Install Python Dependencies:** Ensure you have **Python 3.10+** (the tool is tested with Python 3.12 ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=python))). Install the required libraries using pip:  
   ```bash
   pip install -r requirements.txt
   ```  
   This will install packages such as `requests`, `python-dotenv`, `PyMuPDF`, `pathlib`, `tqdm`, and `click` ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=requests)) which are needed for the optimizer to run.

3. **Set Up API Keys:** The optimizer uses external AI services, so you need API keys for **DeepSeek** and **Perplexity**.  
   - Create a file named `.env` in the project root (or use the provided `.env` template) and add your keys:  
     ```dotenv
     DEEPSEEK_API_KEY=<your_deepseek_api_key>  
     PERPLEXITY_API_KEY=<your_perplexity_api_key>
     ```  
   Replace `<your_deepseek_api_key>` and `<your_perplexity_api_key>` with your actual credentials. (Sign up at the DeepSeek platform to obtain an API key, and ensure you have access to Perplexity's API.) The program will automatically load these from the environment on startup.  

4. **(Optional) CUDA Toolkit:** If you plan to compile and benchmark the optimized kernels, make sure the NVIDIA CUDA Toolkit (including `nvcc`) is installed. The optimizer itself does not require compiling code, but having CUDA installed allows you to compile/run the results to verify performance.

## Usage Guide

Once installed and configured, you can run the CUDA Optimizer via the command-line interface. The main entry point is the `driver.py` script which accepts an input CUDA file and outputs optimized kernel code:

```bash
python driver.py --input-file path/to/your_kernel.cu --output-dir output_folder -k 3 -v
```  

**Parameters:**  
- `-i, --input-file` – Path to the CUDA C/C++ source file containing the kernel you want to optimize ([cuda-opt/driver.py at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/driver.py#:~:text=%40click)). For example, you can use the sample provided in `tests/matmul.cu` as a starting point.  
- `-o, --output-dir` – Directory where results will be saved (it will be created if it doesn’t exist). The tool will produce output files in this folder.  
- `-k, --kernels` – (Optional) Number of optimized kernel variants to generate. Default is 3 ([cuda-opt/driver.py at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/driver.py#:~:text=%40click.option%28%27)). Increase or decrease this to get more or fewer suggestions.  
- `-v, --verbose` – (Optional) Enable verbose logging. This will display detailed logs of each step, including paper searches and API interactions.

**Example:** Suppose you have a CUDA kernel file `my_kernel.cu`. To generate 3 optimized versions of this kernel, run:  
```bash
python driver.py -i my_kernel.cu -o optim_results -k 3 -v
```  
This command will create an `optim_results/` directory (if not existing) with a subfolder `candidate_kernels/` containing files like `candidate_1.cu`, `candidate_2.cu`, etc., each being an AI-optimized variant of your original kernel. It will also log the process (showing steps like "Reading input CUDA kernel...", "Searching for relevant papers...", and "Saving optimized kernels...") to the console and to a log file in the output directory. You can open and review the generated `.cu` files to see the suggested optimizations.

## Dependencies

To run Cuda Optimizer, you need the following software and libraries:

- **Python 3.10 or higher** – The code is written in Python and makes use of features compatible with 3.10+. (Python 3.12 is confirmed to work ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=python)).)  
- **CUDA Toolkit (optional)** – NVIDIA’s CUDA toolkit is only required if you want to compile or test the output kernels on GPU hardware. It's not required just to generate the code.  
- **Python Libraries:** The required Python packages are listed in `requirements.txt` and include:  
  - `requests` – for making API calls to DeepSeek and Perplexity ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=requests))  
  - `python-dotenv` – to load API keys from the `.env` file ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=requests))  
  - `PyMuPDF` – for parsing PDF files (used to extract content from research papers) ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=python))  
  - `pathlib` – for convenient file path manipulations ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=PyMuPDF%3D%3D1.22.5%20,12))  
  - `tqdm` – for progress bars during processing (if applicable) ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=PyMuPDF%3D%3D1.22.5%20,12))  
  - `click` – for the command-line interface framework ([cuda-opt/requirements.txt at main · sathvikr/cuda-opt · GitHub](https://github.com/sathvikr/cuda-opt/blob/main/requirements.txt#:~:text=tqdm))  
  These will be installed automatically via the installation step above.

- **Internet Connection:** An active internet connection is required at runtime. The optimizer will query external services (ArXiv via the Perplexity API, and DeepSeek’s model inference API) to gather optimization insights and generate code. Ensure your environment can access the internet when running the tool.

## Contributors

This project is a collaboration between multiple contributors on GitHub ([GitHub - sathvikr/cuda-opt: AI-generated CUDA kernels.](https://github.com/sathvikr/cuda-opt#:~:text=Contributors%203)):

- **Sathvik R.** ([@sathvikr](https://github.com/sathvikr)) – Creator/Maintainer  
- **Leo Nagel** ([@leonagel](https://github.com/leonagel)) – Contributor (development and optimization research)  
- **Rishi** ([@rishipython](https://github.com/rishipython)) – Contributor (development and testing)

Contributions from the community are welcome. If you encounter issues or have ideas for improvements, feel free to open an issue or submit a pull request. Please ensure any contributions align with the project goals and that you have the rights to any code you contribute.

## License

*As of now, no explicit license is specified for this repository.* This means the project is **not currently under a standard open-source license**. All rights to the source code are reserved to the authors by default. If you intend to use this code beyond personal experimentation, please check for any updates to the licensing or contact the maintainers for clarification. (The repository may choose to add an open-source license in the future, so it's a good idea to look for a `LICENSE` file or notice in the repository for any updates.)
