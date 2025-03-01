import requests
import json
import fitz
import os

PERPLEXITY_API_KEY = "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"

def search_arxiv_via_perplexity(cuda_code, num_results=2, api_key=PERPLEXITY_API_KEY):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are an expert CUDA researcher."},
            {"role": "user", "content": f"Find arxiv papers that will help an AI optimize the following CUDA kernel: {cuda_code}"}],
        "max_tokens": 500,
        "search_domain_filter": ["arxiv.org"],
        "num_results": num_results,
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        urls = data.get("citations", [])
        return urls
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def extract_arxiv_text(arxiv_url, save_dir="papers2"): 
    # Ensure the URL is from arXiv
    if "arxiv.org" not in arxiv_url:
        print(f"Skipping non-ArXiv URL: {arxiv_url}")
        return ""
    
    # Convert to PDF URL if necessary
    if "/abs/" in arxiv_url or "/html/" in arxiv_url:
        paper_id = arxiv_url.split("/")[-1].split("v")[0]  # Extracts ID, ignoring versioning
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    elif "/pdf/" in arxiv_url:
        pdf_url = arxiv_url  # Already a PDF link
    else:
        print(f"Unrecognized ArXiv format: {arxiv_url}")
        return ""
    
    pdf_path = os.path.join(save_dir, f"{pdf_url.split('/')[-1]}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading paper: {pdf_url}")
    response = requests.get(pdf_url, stream=True)
    if response.status_code == 200:
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"Saved PDF: {pdf_path}")
    else:
        print("Error downloading PDF.")
        return ""
    
    # Extract text from PDF
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
    
def summarize_arxiv_text_with_deepseek(text, code=None):
    # api_key = os.getenv("DEEPSEEK_API_KEY")
    api_key = "sk-866ea93edb4d47f092903d62eb295e1d"
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if code is not None: user_prompt = f"I want to optimize this CUDA code: {code}. Extract the relevant information from the following paper, providing a detailed and concise summary with all useful information: {text}."
    else: user_prompt = f"I want to optimize CUDA code. Extract the relevant information from the following paper, providing a detailed and concise summary with all useful information: {text}."
    
    # Prepare the request payload
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": user_prompt}]
    }

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"]
            return summary
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise
    except Exception as e:
        raise

if __name__ == "__main__":
    from loader import read_cu, write_cu

    input_file = "tests/matmul.cu"
    output_file = "tests/matmul_opt.cu"
    
    code = read_cu(input_file)
    
    urls = search_arxiv_via_perplexity(code)
    print(urls)
    summaries = list(map(
        lambda url: extract_arxiv_text(url),
        urls
    ))

    to_output_file = ""
    for summary in summaries:
        to_output_file = summary + "\n"
    
    write_cu(output_file, to_output_file)