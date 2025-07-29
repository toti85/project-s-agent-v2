#!/usr/bin/env python3
"""
Project-S V2 - Web Scraping & Browser Automation Tools
=====================================================

Professional web scraping and browser automation tools for Project-S V2.
Built from the archaeological findings of the Golden Age web_scraper.py.

Author: Project-S V2 AI System
Date: 2025-07-01
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
import csv
import os
import httpx
import urllib.parse


class WebScraperTool:
    """
    Professional web scraping tool with advanced features
    """
    
    def __init__(self):
        self.name = "web_scraper"
        self.description = "Advanced web scraping tool for extracting data from websites"
        self.session = None
        self.results = []
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
    async def execute(self, parameters: Dict) -> Dict:
        """
        Execute web scraping operation
        
        Parameters:
        - url: Target URL to scrape
        - selectors: Dict of CSS selectors for data extraction
        - output_format: 'json', 'csv', 'txt'
        - delay: Delay between requests (seconds)
        - max_pages: Maximum pages to scrape
        """
        try:
            url = parameters.get('url')
            selectors = parameters.get('selectors', {})
            output_format = parameters.get('output_format', 'json')
            delay = parameters.get('delay', 1)
            max_pages = parameters.get('max_pages', 1)
            
            if not url:
                return {"success": False, "error": "URL parameter is required"}
            
            # Initialize session
            await self._init_session()
            
            # Scrape data
            scraped_data = await self._scrape_url(url, selectors)
            
            if scraped_data:
                # Save results
                output_file = await self._save_results(scraped_data, output_format)
                
                await self._cleanup_session()
                
                return {
                    "success": True,
                    "data": scraped_data,
                    "output_file": output_file,
                    "items_scraped": len(scraped_data) if isinstance(scraped_data, list) else 1,
                    "message": f"Successfully scraped {len(scraped_data) if isinstance(scraped_data, list) else 1} items"
                }
            else:
                await self._cleanup_session()
                return {"success": False, "error": "No data could be scraped from the URL"}
                
        except Exception as e:
            await self._cleanup_session()
            return {"success": False, "error": f"Web scraping failed: {str(e)}"}
    
    async def _init_session(self):
        """Initialize aiohttp session"""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    async def _cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _scrape_url(self, url: str, selectors: Dict) -> List[Dict]:
        """Scrape data from a single URL"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract data based on selectors
                    if selectors:
                        return self._extract_with_selectors(soup, selectors, url)
                    else:
                        # Default extraction for common e-commerce patterns
                        return self._extract_default_patterns(soup, url)
                else:
                    print(f"Failed to fetch {url}: HTTP {response.status}")
                    return []
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []
    
    def _extract_with_selectors(self, soup: BeautifulSoup, selectors: Dict, url: str) -> List[Dict]:
        """Extract data using custom CSS selectors"""
        results = []
        
        # Check if we're looking for multiple items or single item
        container_selector = selectors.get('container')
        
        if container_selector:
            # Multiple items
            containers = soup.select(container_selector)
            for container in containers:
                item_data = {"url": url}
                for field, selector in selectors.items():
                    if field != 'container':
                        element = container.select_one(selector)
                        item_data[field] = element.get_text(strip=True) if element else None
                results.append(item_data)
        else:
            # Single item
            item_data = {"url": url}
            for field, selector in selectors.items():
                elements = soup.select(selector)
                if elements:
                    if len(elements) == 1:
                        item_data[field] = elements[0].get_text(strip=True)
                    else:
                        item_data[field] = [el.get_text(strip=True) for el in elements]
                else:
                    item_data[field] = None
            results.append(item_data)
        
        return results
    
    def _extract_default_patterns(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Extract data using common patterns"""
        data = {"url": url, "scraped_at": datetime.now().isoformat()}
        
        # Common patterns for different types of content
        title_selectors = ['h1', 'title', '.title', '#title', '.product-title', '.post-title']
        price_selectors = ['.price', '.cost', '.amount', '[class*="price"]', '[id*="price"]']
        description_selectors = ['.description', '.summary', '.excerpt', 'meta[name="description"]']
        
        # Extract title
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                data['title'] = element.get_text(strip=True)
                break
        
        # Extract price
        for selector in price_selectors:
            element = soup.select_one(selector)
            if element:
                data['price'] = element.get_text(strip=True)
                break
        
        # Extract description
        for selector in description_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    data['description'] = element.get('content', '')
                else:
                    data['description'] = element.get_text(strip=True)
                break
        
        # Extract all links
        links = soup.find_all('a', href=True)
        data['links'] = [urljoin(url, link['href']) for link in links[:10]]  # Limit to 10 links
        
        # Extract all images
        images = soup.find_all('img', src=True)
        data['images'] = [urljoin(url, img['src']) for img in images[:5]]  # Limit to 5 images
        
        return [data]
    
    async def _save_results(self, data: List[Dict], output_format: str) -> str:
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == 'csv':
            filename = f"scraped_data_{timestamp}.csv"
            await self._save_to_csv(data, filename)
        elif output_format.lower() == 'txt':
            filename = f"scraped_data_{timestamp}.txt"
            await self._save_to_txt(data, filename)
        else:  # Default to JSON
            filename = f"scraped_data_{timestamp}.json"
            await self._save_to_json(data, filename)
        
        return filename
    
    async def _save_to_json(self, data: List[Dict], filename: str):
        """Save data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def _save_to_csv(self, data: List[Dict], filename: str):
        """Save data to CSV file"""
        if not data:
            return
        
        # Get all unique keys from all dictionaries
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_keys))
            writer.writeheader()
            writer.writerows(data)
    
    async def _save_to_txt(self, data: List[Dict], filename: str):
        """Save data to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data, 1):
                f.write(f"Item {i}:\n")
                f.write("-" * 40 + "\n")
                for key, value in item.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")


class WebAnalyzerTool:
    """
    Web analysis tool for extracting insights from scraped data
    """
    
    def __init__(self):
        self.name = "web_analyzer"
        self.description = "Analyze scraped web data for patterns and insights"
    
    async def execute(self, parameters: Dict) -> Dict:
        """
        Analyze web data
        
        Parameters:
        - data_file: Path to scraped data file (optional if data is provided)
        - data: Direct data to analyze (optional if data_file is provided)
        - analysis_type: 'content', 'pricing', 'links', 'trends'
        """
        try:
            data_file = parameters.get('data_file')
            direct_data = parameters.get('data')
            analysis_type = parameters.get('analysis_type', 'content')
            
            # Load data from file or use direct data
            if direct_data:
                data = direct_data
            elif data_file and os.path.exists(data_file):
                data = await self._load_data(data_file)
            else:
                return {"success": False, "error": "Either data_file or data parameter is required"}
            
            if not data:
                return {"success": False, "error": "No data found in file"}
            
            # Perform analysis
            analysis_result = await self._analyze_data(data, analysis_type)
            
            # Save analysis report
            report_file = await self._save_analysis_report(analysis_result, analysis_type)
            
            return {
                "success": True,
                "analysis": analysis_result,
                "report_file": report_file,
                "items_analyzed": len(data)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Web analysis failed: {str(e)}"}
    
    async def _load_data(self, data_file: str) -> List[Dict]:
        """Load data from file"""
        if data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif data_file.endswith('.csv'):
            with open(data_file, 'r', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        else:
            return []
    
    async def _analyze_data(self, data: List[Dict], analysis_type: str) -> Dict:
        """Perform data analysis"""
        analysis = {
            "analysis_type": analysis_type,
            "total_items": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
        if analysis_type == 'content':
            analysis.update(self._analyze_content(data))
        elif analysis_type == 'pricing':
            analysis.update(self._analyze_pricing(data))
        elif analysis_type == 'links':
            analysis.update(self._analyze_links(data))
        elif analysis_type == 'trends':
            analysis.update(self._analyze_trends(data))
        
        return analysis
    
    def _analyze_content(self, data: List[Dict]) -> Dict:
        """Analyze content patterns"""
        titles = [item.get('title', '') for item in data if item.get('title')]
        descriptions = [item.get('description', '') for item in data if item.get('description')]
        
        return {
            "title_count": len(titles),
            "description_count": len(descriptions),
            "avg_title_length": sum(len(t) for t in titles) / len(titles) if titles else 0,
            "avg_description_length": sum(len(d) for d in descriptions) / len(descriptions) if descriptions else 0,
            "common_words": self._get_common_words(titles + descriptions)
        }
    
    def _analyze_pricing(self, data: List[Dict]) -> Dict:
        """Analyze pricing data"""
        prices = []
        for item in data:
            price_text = item.get('price', '')
            if price_text:
                # Extract numeric value from price string
                import re
                numbers = re.findall(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                if numbers:
                    try:
                        prices.append(float(numbers[0]))
                    except ValueError:
                        pass
        
        if prices:
            return {
                "price_count": len(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "price_range": max(prices) - min(prices)
            }
        else:
            return {"price_count": 0, "message": "No valid pricing data found"}
    
    def _analyze_links(self, data: List[Dict]) -> Dict:
        """Analyze link patterns"""
        all_links = []
        for item in data:
            links = item.get('links', [])
            if isinstance(links, list):
                all_links.extend(links)
        
        domains = [urlparse(link).netloc for link in all_links]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            "total_links": len(all_links),
            "unique_domains": len(set(domains)),
            "top_domains": sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _analyze_trends(self, data: List[Dict]) -> Dict:
        """Analyze trending patterns"""
        # Basic trend analysis - can be enhanced
        return {
            "data_points": len(data),
            "has_timestamps": any('scraped_at' in item for item in data),
            "unique_sources": len(set(item.get('url', '') for item in data))
        }
    
    def _get_common_words(self, texts: List[str], top_n: int = 10) -> List[tuple]:
        """Get most common words from texts"""
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    async def _save_analysis_report(self, analysis: Dict, analysis_type: str) -> str:
        """Save analysis report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_analysis_{analysis_type}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        return filename


class WebPageFetchTool:
    """
    Weboldal letöltése és tartalom kinyerése - LEGACY COMPATIBILITY
    
    Category: web
    Version: 1.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    def __init__(self):
        self.name = "web_page_fetch"
        self.description = "Fetch web page content and extract text"
    
    async def execute(self, 
                    url: str, 
                    extract_text: bool = True,
                    timeout: int = 10,
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Letölt egy weboldalt és visszaadja a tartalmát.
        
        Args:
            url: A weboldal URL-je
            extract_text: Ha True, kinyeri a szöveges tartalmat a HTML-ből
            timeout: Időtúllépés másodpercben
            headers: Opcionális HTTP fejlécek
            
        Returns:
            Dict: Az eredmény szótár formában
        """
        try:
            # URL ellenőrzés
            if not url.startswith(("http://", "https://")):
                return {
                    "success": False,
                    "error": f"Érvénytelen URL: {url} - Az URL-nek http:// vagy https://-sel kell kezdődnie"
                }
                
            # Alapértelmezett fejlécek
            default_headers = {
                "User-Agent": "Project-S Agent/1.0 (compatible; ModernWebKit)"
            }
            
            # Fejlécek egyesítése, ha vannak megadva egyedi fejlécek
            if headers:
                default_headers.update(headers)
                
            # Weboldal letöltése
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=default_headers)
                
                # Státusz kód ellenőrzés
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"A kérés sikertelen, státusz kód: {response.status_code}",
                        "status_code": response.status_code
                    }
                    
                # Szöveges tartalom kinyerése, ha kérték
                text_content = None
                if extract_text:
                    # HTML elemzés BeautifulSoup-pal
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Script és stílus elemek eltávolítása
                    for script_or_style in soup(["script", "style"]):
                        script_or_style.extract()
                        
                    # Szöveges tartalom kinyerése
                    text_content = soup.get_text(separator='\n', strip=True)
                    
                # Metaadatok kinyerése
                title = None
                description = None
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Cím keresése
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.text.strip()
                    
                # Leírás keresése
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    description = meta_desc["content"].strip()
                    
                # Eredmény összeállítása
                result = {
                    "success": True,
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type", "unknown"),
                    "title": title,
                    "description": description,
                    "html": response.text
                }
                
                if extract_text:
                    result["text"] = text_content
                    
                return result
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"Időtúllépés a weboldal letöltése közben: {url}"
            }
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": f"Hiba a kérés végrehajtása közben: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


class WebApiCallTool:
    """
    HTTP API hívások végrehajtása - LEGACY COMPATIBILITY
    
    Category: web
    Version: 1.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    def __init__(self):
        self.name = "web_api_call"
        self.description = "Execute HTTP API calls"
    
    async def execute(self, 
                    url: str, 
                    method: str = "GET",
                    headers: Optional[Dict[str, str]] = None,
                    data: Optional[Union[Dict[str, Any], str]] = None,
                    params: Optional[Dict[str, Any]] = None,
                    timeout: int = 10,
                    json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        HTTP API hívás végrehajtása.
        
        Args:
            url: Az API végpont URL-je
            method: HTTP metódus (GET, POST, PUT, DELETE, stb.)
            headers: HTTP fejlécek
            data: Küldendő adatok (form data vagy raw string)
            params: URL paraméterek
            timeout: Időtúllépés másodpercben
            json_data: JSON adat, ha van
            
        Returns:
            Dict: Az API hívás eredménye
        """
        try:
            # URL ellenőrzés
            if not url.startswith(("http://", "https://")):
                return {
                    "success": False,
                    "error": f"Érvénytelen URL: {url} - Az URL-nek http:// vagy https://-sel kell kezdődnie"
                }
                
            # HTTP metódus ellenőrzése
            method = method.upper()
            if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
                return {
                    "success": False,
                    "error": f"Érvénytelen HTTP metódus: {method}"
                }
                
            # Alapértelmezett fejlécek
            default_headers = {
                "User-Agent": "Project-S Agent/1.0 (API Client)"
            }
            
            # Fejlécek egyesítése, ha vannak megadva egyedi fejlécek
            if headers:
                default_headers.update(headers)
                
            # API hívás végrehajtása
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.request(
                    method=method, 
                    url=url, 
                    headers=default_headers,
                    params=params,
                    data=data,
                    json=json_data
                )
                
                # Eredmény összeállítása
                result = {
                    "success": True,
                    "url": url,
                    "method": method,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                }
                
                # Próbáljuk JSON-ként értelmezni, ha lehetséges
                try:
                    result["response"] = response.json()
                    result["content_type"] = "application/json"
                except:
                    result["response"] = response.text
                    result["content_type"] = response.headers.get("Content-Type", "text/plain")
                    
                return result
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"Időtúllépés az API hívás közben: {url}"
            }
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": f"Hiba a kérés végrehajtása közben: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


class WebSearchTool:
    """
    Webes keresés végrehajtása - LEGACY COMPATIBILITY
    
    Category: web
    Version: 1.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Perform web search using Google"
    
    async def execute(self, 
                    query: str, 
                    max_results: int = 5) -> Dict[str, Any]:
        """
        Webes keresés végrehajtása.
        
        Args:
            query: A keresési lekérdezés
            max_results: Maximálisan visszaadott találatok száma
            
        Returns:
            Dict: A keresési eredmények
        """
        try:
            # Lekérdezés URL-kódolása
            encoded_query = urllib.parse.quote_plus(query)
            
            # Összeállítjuk a keresési URL-t (egyszerű Google keresés)
            search_url = f"https://www.google.com/search?q={encoded_query}&num={max_results + 5}"
            
            # Fejlécek (böngésző utánzás)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            # Keresés végrehajtása
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                response = await client.get(search_url, headers=headers)
                
                # Státusz kód ellenőrzés
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"A keresési kérés sikertelen, státusz kód: {response.status_code}"
                    }
                    
                # HTML elemzés
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Keresési eredmények kinyerése
                search_results = []
                for result in soup.select("div.g"):
                    # Cím és URL kinyerése
                    title_element = result.select_one("h3")
                    link_element = result.select_one("a")
                    
                    if title_element and link_element and "href" in link_element.attrs:
                        link = link_element["href"]
                        
                        # Csak valódi linkek, nem Google belső linkek
                        if link.startswith("http") and "google.com" not in link:
                            # Leírás kinyerése
                            description = ""
                            desc_element = result.select_one("div.VwiC3b")
                            if desc_element:
                                description = desc_element.text.strip()
                                
                            search_results.append({
                                "title": title_element.text.strip(),
                                "url": link,
                                "description": description
                            })
                            
                            # Ha elérjük a maximális eredményszámot, kilépünk
                            if len(search_results) >= max_results:
                                break
                                
                return {
                    "success": True,
                    "query": query,
                    "results": search_results,
                    "count": len(search_results)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


# Tool registry integration
AVAILABLE_TOOLS = {
    "web_scraper": WebScraperTool,
    "web_analyzer": WebAnalyzerTool,
    "web_page_fetch": WebPageFetchTool,
    "web_api_call": WebApiCallTool,
    "web_search": WebSearchTool
}

if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("🌐 Project-S V2 Web Tools Demo")
        print("=" * 50)
        
        # Demo scraping
        scraper = WebScraperTool()
        result = await scraper.execute({
            'url': 'https://example.com',
            'output_format': 'json'
        })
        
        print(f"Scraping result: {result}")
    
    asyncio.run(demo())
