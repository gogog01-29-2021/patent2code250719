import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time
import logging
from datetime import datetime, timedelta

class PatentDataCollector:
    """특허 데이터 수집기"""
    
    def __init__(self):
        self.uspto_base_url = "https://developer.uspto.gov/ds-api"
        self.epo_base_url = "https://ops.epo.org/3.2/rest-services"
        self.google_patents_url = "https://patents.googleapis.com/v1/patents"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Patent2Tech/1.0 (contact@patent2tech.com)'
        })
    
    def search_uspto_patents(self, query: str, limit: int = 100) -> List[Dict]:
        """USPTO에서 특허 검색"""
        try:
            # USPTO API 사용 (실제 구현)
            params = {
                'q': query,
                'f': 'json',
                'o': 'relevance',
                's': limit
            }
            
            response = self.session.get(
                f"{self.uspto_base_url}/search/patents",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            patents = []
            
            for item in data.get('patents', []):
                patent = {
                    'patent_number': item.get('patentNumber'),
                    'title': item.get('title'),
                    'abstract': item.get('abstract'),
                    'inventors': item.get('inventors', []),
                    'assignees': item.get('assignees', []),
                    'filing_date': item.get('filingDate'),
                    'publication_date': item.get('publicationDate'),
                    'patent_office': 'USPTO',
                    'source_url': item.get('url')
                }
                patents.append(patent)
            
            return patents
            
        except Exception as e:
            logging.error(f"Error searching USPTO patents: {e}")
            return []
    
    def get_patent_details(self, patent_number: str, office: str = 'USPTO') -> Optional[Dict]:
        """특허 상세 정보 조회"""
        try:
            if office == 'USPTO':
                url = f"{self.uspto_base_url}/patents/{patent_number}"
            elif office == 'EPO':
                url = f"{self.epo_base_url}/published-data/publication/epodoc/{patent_number}"
            else:
                return None
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if office == 'USPTO':
                data = response.json()
                return {
                    'claims': data.get('claims'),
                    'description': data.get('description'),
                    'figures': data.get('figures', []),
                    'citations': data.get('citations', []),
                    'classification_codes': data.get('classifications', [])
                }
            
        except Exception as e:
            logging.error(f"Error getting patent details: {e}")
            return None
    
    def download_patent_figures(self, patent_number: str, figures_data: List[Dict]) -> List[str]:
        """특허 도면 다운로드"""
        downloaded_files = []
        
        for i, figure in enumerate(figures_data):
            try:
                figure_url = figure.get('url')
                if not figure_url:
                    continue
                
                response = self.session.get(figure_url, timeout=30)
                response.raise_for_status()
                
                # 파일 저장
                filename = f"patents/{patent_number}/figure_{i+1}.png"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(filename)
                time.sleep(1)  # API 제한 고려
                
            except Exception as e:
                logging.error(f"Error downloading figure {i+1}: {e}")
        
        return downloaded_files