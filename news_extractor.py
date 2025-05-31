# news_extractor.py
"""
Module untuk ekstraksi konten berita dari berbagai situs berita 
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime


class NewsExtractor:
    """Kelas untuk mengekstrak judul dan konten dari URL berita"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_from_url(self, url):
        """Extract title and content from news URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse domain to apply specific extraction rules
            domain = urlparse(url).netloc
            
            # Initialize results
            title = ""
            content = ""
            
            # Domain-specific extraction rules
            if 'detik.com' in domain:
                title, content = self._extract_detik(soup)
            elif 'kompas.com' in domain:
                title, content = self._extract_kompas(soup)
            elif 'tribunnews.com' in domain:
                title, content = self._extract_tribun(soup)
            elif 'cnnindonesia.com' in domain:
                title, content = self._extract_cnn(soup)
            elif 'liputan6.com' in domain:
                title, content = self._extract_liputan6(soup)
            else:
                # Generic extraction
                title, content = self._extract_generic(soup)
            
            return {
                'success': True,
                'title': title,
                'content': content,
                'url': url,
                'domain': domain,
                'extracted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_detik(self, soup):
        """Extract from detik.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='detail__title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='detail__body-text') or soup.find('div', class_='itp_bodycontent')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_kompas(self, soup):
        """Extract from kompas.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='read__title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='read__content') or soup.find('div', class_='content')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_tribun(self, soup):
        """Extract from tribunnews.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', id='arttitle') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='content') or soup.find('div', class_='txt-article')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_cnn(self, soup):
        """Extract from cnnindonesia.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', id='detikdetailtext') or soup.find('div', class_='detail-text')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_liputan6(self, soup):
        """Extract from liputan6.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='read-page--header--title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='article-content-body__item-content') or soup.find('div', class_='article-content')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_generic(self, soup):
        """Generic extraction for unknown domains"""
        title = ""
        content = ""
        
        # Title extraction - try multiple selectors
        title_selectors = [
            'h1',
            'title',
            'meta[property="og:title"]',
            'meta[name="twitter:title"]'
        ]
        
        for selector in title_selectors:
            elem = soup.find(selector)
            if elem:
                if selector.startswith('meta'):
                    title = elem.get('content', ' ')
                else:
                    title = elem.get_text()
                title = title.strip()
                if title:
                    break
        
        # Content extraction - try to find article body
        content_selectors = [
            'article',
            'div[class*="content"]',
            'div[class*="article"]',
            'div[class*="post"]',
            'main'
        ]
        
        for selector in content_selectors:
            if '[' in selector:
                # Handle attribute selectors
                tag, attr = selector.split('[')
                attr = attr.rstrip(']')
                key, value = attr.split('*=') 
                elem = soup.find(tag, class_=lambda x: x and value.strip('"') in x)
            else:
                elem = soup.find(selector)
            
            if elem:
                paragraphs = elem.find_all('p')
                if paragraphs:
                    content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    if content:
                        break
        
        return title, content