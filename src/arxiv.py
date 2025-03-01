import requests
import json
import fitz
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ArxivClient:
    def __init__(self, perplexity_api_key: str, papers_dir: str = "papers2"):
        self.perplexity_api_key = perplexity_api_key
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(exist_ok=True)

    def search_papers(self, cuda_code: str, num_results: int = 3) -> list[str]:
        """Search for relevant ArXiv papers using Perplexity API"""
        logger.info("🔍 Searching for relevant papers on ArXiv...")
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "You are an expert CUDA researcher."},
                {"role": "user", "content": f"Find arxiv papers that will help an AI optimize the following CUDA kernel: {cuda_code}"}
            ],
            "max_tokens": 500,
            "search_domain_filter": ["arxiv.org"],
            "num_results": num_results,
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            # Ensure we only return the requested number of papers
            papers = response.json().get("citations", [])
            papers = [p for p in papers if "arxiv.org" in p]  # Filter non-arxiv papers
            return papers[:num_results]  # Limit to requested number
        except Exception as e:
            logger.error(f"Error searching ArXiv papers: {e}")
            return []

    def extract_paper_text(self, arxiv_url: str) -> str:
        """Download and extract text from ArXiv paper"""
        if "arxiv.org" not in arxiv_url:
            logger.warning(f"⚠️ Skipping non-ArXiv URL: {arxiv_url}")
            return ""
        
        # Convert to PDF URL if necessary
        if "/abs/" in arxiv_url or "/html/" in arxiv_url:
            paper_id = arxiv_url.split("/")[-1].split("v")[0]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        elif "/pdf/" in arxiv_url:
            pdf_url = arxiv_url
        else:
            logger.warning(f"⚠️ Unrecognized ArXiv format: {arxiv_url}")
            return ""
        
        pdf_path = self.papers_dir / f"{pdf_url.split('/')[-1]}"
        
        try:
            # Download PDF without progress bar for individual files
            logger.info(f"📥 Downloading paper: {pdf_url.split('/')[-1]}")
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            
            # Extract text with logging
            logger.info(f"📄 Extracting text from PDF...")
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])
            logger.info(f"✓ Successfully processed paper ({len(text)} chars extracted)")
            return text
            
        except Exception as e:
            logger.error(f"❌ Error processing paper {arxiv_url}: {e}")
            return ""
