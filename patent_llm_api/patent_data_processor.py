#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특허 데이터 처리 및 전처리 모듈
Patent Data Processing and Preprocessing Module

특허 문서의 텍스트, 도면, 메타데이터를 처리하고 LLM 훈련에 적합한 형태로 변환

MIT License
Copyright (c) 2025
"""

import os
import re
import json
import sqlite3
import requests
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path
import base64
from datetime import datetime
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import cv2

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessedPatentData:
    """처리된 특허 데이터 클래스"""
    patent_id: str
    processed_text: str
    structured_claims: List[Dict]
    extracted_figures: List[Dict]
    technical_keywords: List[str]
    classification_hierarchy: Dict
    processing_metadata: Dict

class PatentTextProcessor:
    """특허 텍스트 전처리기"""
    
    def __init__(self):
        # 특허 전용 정규표현식 패턴
        self.claim_pattern = r'(?:Claim|CLAIM)\s*(\d+)\.?\s*(.+?)(?=(?:Claim|CLAIM)\s*\d+|$)'
        self.figure_ref_pattern = r'(?:FIG\.|Figure|Fig\.)\s*(\d+[A-Za-z]?)'
        self.patent_id_pattern = r'(?:US|EP|WO|KR|JP|CN)\s*\d+\s*[A-Z]\d*'
        
        # 기술 분야별 키워드 사전
        self.tech_keywords = {
            'ai_ml': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'algorithm'],
            'electronics': ['circuit', 'semiconductor', 'transistor', 'voltage', 'current', 'resistance'],
            'mechanical': ['mechanism', 'gear', 'bearing', 'actuator', 'motor', 'transmission'],
            'chemical': ['compound', 'reaction', 'catalyst', 'synthesis', 'polymer', 'molecule'],
            'biotech': ['protein', 'gene', 'dna', 'enzyme', 'cell', 'biological'],
            'software': ['software', 'program', 'code', 'application', 'system', 'method']
        }
    
    def clean_patent_text(self, raw_text: str) -> str:
        """특허 텍스트 정리"""
        # 불필요한 문자 제거
        text = re.sub(r'\s+', ' ', raw_text)  # 연속 공백 제거
        text = re.sub(r'\n+', '\n', text)     # 연속 줄바꿈 제거
        text = re.sub(r'[^\w\s\.\,\;\:\(\)\[\]\-\+\=\<\>\/\\\"\']', '', text)  # 특수문자 정리
        
        # 특허 특유의 표현 정리
        text = re.sub(r'(?i)background\s+of\s+the\s+invention', 'BACKGROUND', text)
        text = re.sub(r'(?i)summary\s+of\s+the\s+invention', 'SUMMARY', text)
        text = re.sub(r'(?i)detailed\s+description', 'DETAILED_DESCRIPTION', text)
        text = re.sub(r'(?i)brief\s+description\s+of\s+the\s+drawings', 'DRAWINGS_DESCRIPTION', text)
        
        return text.strip()
    
    def extract_claims(self, patent_text: str) -> List[Dict]:
        """특허 클레임 추출 및 구조화"""
        claims = []
        
        # 클레임 섹션 찾기
        claim_section_match = re.search(r'(?i)(?:claims?|what\s+is\s+claimed)\s*:?\s*(.*?)(?:abstract|drawings|$)', 
                                      patent_text, re.DOTALL)
        
        if claim_section_match:
            claim_text = claim_section_match.group(1)
            
            # 개별 클레임 추출
            claim_matches = re.finditer(self.claim_pattern, claim_text, re.DOTALL | re.IGNORECASE)
            
            for match in claim_matches:
                claim_num = int(match.group(1))
                claim_content = match.group(2).strip()
                
                # 클레임 타입 분류
                claim_type = self._classify_claim_type(claim_content)
                
                # 종속 클레임 관계 파악
                dependency = self._extract_claim_dependency(claim_content)
                
                claims.append({
                    'claim_number': claim_num,
                    'content': claim_content,
                    'type': claim_type,
                    'dependency': dependency,
                    'length': len(claim_content.split()),
                    'technical_elements': self._extract_technical_elements(claim_content)
                })
        
        return sorted(claims, key=lambda x: x['claim_number'])
    
    def _classify_claim_type(self, claim_content: str) -> str:
        """클레임 타입 분류"""
        content_lower = claim_content.lower()
        
        if any(keyword in content_lower for keyword in ['method', 'process', 'step', 'procedure']):
            return 'method'
        elif any(keyword in content_lower for keyword in ['system', 'apparatus', 'device', 'machine']):
            return 'apparatus'
        elif any(keyword in content_lower for keyword in ['composition', 'compound', 'material']):
            return 'composition'
        else:
            return 'other'
    
    def _extract_claim_dependency(self, claim_content: str) -> Optional[int]:
        """종속 클레임 관계 추출"""
        dependency_match = re.search(r'(?:according\s+to|of)\s+claim\s+(\d+)', claim_content, re.IGNORECASE)
        if dependency_match:
            return int(dependency_match.group(1))
        return None
    
    def _extract_technical_elements(self, claim_content: str) -> List[str]:
        """기술적 요소 추출"""
        elements = []
        
        # 명사구 추출 (간단한 패턴)
        noun_phrases = re.findall(r'\b(?:[A-Z][a-z]+\s+)*[A-Z][a-z]+\b', claim_content)
        elements.extend(noun_phrases)
        
        # 기술 키워드 매칭
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                if keyword.lower() in claim_content.lower():
                    elements.append(f"{category}:{keyword}")
        
        return list(set(elements))
    
    def extract_technical_keywords(self, patent_text: str) -> List[str]:
        """기술 키워드 추출"""
        keywords = []
        text_lower = patent_text.lower()
        
        # 기술 분야별 키워드 추출
        for category, keyword_list in self.tech_keywords.items():
            for keyword in keyword_list:
                if keyword in text_lower:
                    keywords.append(keyword)
        
        # TF-IDF 기반 중요 단어 추출 (간단한 구현)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', patent_text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도 기반 상위 키워드 추출
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        keywords.extend([word for word, freq in frequent_words if freq > 2])
        
        return list(set(keywords))
    
    def structure_patent_sections(self, patent_text: str) -> Dict[str, str]:
        """특허 섹션 구조화"""
        sections = {}
        
        # 주요 섹션 패턴
        section_patterns = {
            'title': r'(?i)(?:title|invention\s+title)\s*:?\s*(.+?)(?:\n|$)',
            'abstract': r'(?i)abstract\s*:?\s*(.*?)(?=background|field|summary|$)',
            'background': r'(?i)background\s*(?:of\s+the\s+invention)?\s*:?\s*(.*?)(?=summary|field|detailed|$)',
            'summary': r'(?i)summary\s*(?:of\s+the\s+invention)?\s*:?\s*(.*?)(?=detailed|drawings|claims|$)',
            'detailed_description': r'(?i)detailed\s+description\s*:?\s*(.*?)(?=claims|drawings|$)',
            'claims': r'(?i)claims?\s*:?\s*(.*?)(?=abstract|drawings|$)'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, patent_text, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        return sections

class PatentFigureProcessor:
    """특허 도면 처리기"""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf']
        
    def extract_figures_from_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """PDF에서 도면 추출"""
        figure_paths = []
        
        try:
            # PDF에서 이미지 추출 (pdf2image 라이브러리 사용 가정)
            import pdf2image
            
            pages = pdf2image.convert_from_path(pdf_path)
            
            for i, page in enumerate(pages):
                figure_path = os.path.join(output_dir, f"figure_{i+1}.png")
                page.save(figure_path, 'PNG')
                figure_paths.append(figure_path)
                
        except ImportError:
            logger.warning("pdf2image 라이브러리가 설치되지 않았습니다.")
        except Exception as e:
            logger.error(f"PDF 도면 추출 오류: {e}")
        
        return figure_paths
    
    def analyze_figure_content(self, image_path: str) -> Dict:
        """도면 내용 분석"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "이미지 로드 실패"}
            
            # 이미지 기본 정보
            height, width = image.shape[:2]
            
            # OCR을 통한 텍스트 추출
            text = pytesseract.image_to_string(Image.open(image_path))
            
            # 도면 요소 분석
            analysis = {
                "image_path": image_path,
                "dimensions": {"width": width, "height": height},
                "extracted_text": text.strip(),
                "figure_numbers": self._extract_figure_numbers(text),
                "reference_numerals": self._extract_reference_numerals(text),
                "diagram_type": self._classify_diagram_type(image, text),
                "complexity_score": self._calculate_complexity_score(image)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"도면 분석 오류: {e}")
            return {"error": str(e)}
    
    def _extract_figure_numbers(self, text: str) -> List[str]:
        """도면 번호 추출"""
        figure_numbers = re.findall(r'(?:FIG\.|Figure|Fig\.)\s*(\d+[A-Za-z]?)', text, re.IGNORECASE)
        return list(set(figure_numbers))
    
    def _extract_reference_numerals(self, text: str) -> List[str]:
        """참조 번호 추출"""
        # 일반적으로 특허 도면에서 사용되는 참조 번호 패턴
        numerals = re.findall(r'\b(\d{1,3})\b', text)
        return [num for num in set(numerals) if 10 <= int(num) <= 999]
    
    def _classify_diagram_type(self, image: np.ndarray, text: str) -> str:
        """도면 타입 분류"""
        text_lower = text.lower()
        
        # 텍스트 기반 분류
        if any(keyword in text_lower for keyword in ['circuit', 'schematic', 'electrical']):
            return 'circuit_diagram'
        elif any(keyword in text_lower for keyword in ['flow', 'process', 'method']):
            return 'flowchart'
        elif any(keyword in text_lower for keyword in ['block', 'system', 'architecture']):
            return 'block_diagram'
        
        # 이미지 기반 분류 (간단한 휴리스틱)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 직선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        # 원 검출
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=100)
        circle_count = len(circles[0]) if circles is not None else 0
        
        if line_count > 20 and circle_count > 5:
            return 'technical_drawing'
        elif line_count > 10:
            return 'schematic'
        else:
            return 'illustration'
    
    def _calculate_complexity_score(self, image: np.ndarray) -> float:
        """도면 복잡도 점수 계산"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 엣지 밀도 계산
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 텍스처 복잡도 (표준편차 기반)
        texture_complexity = np.std(gray) / 255.0
        
        # 전체 복잡도 점수 (0-1 범위)
        complexity_score = (edge_density * 0.7 + texture_complexity * 0.3)
        
        return min(complexity_score, 1.0)

class PatentDataCollector:
    """특허 데이터 수집기"""
    
    def __init__(self, data_dir: str = "patent_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 데이터 소스 설정
        self.data_sources = {
            'uspto': {
                'bulk_data_url': 'https://data.uspto.gov/bulkdata',
                'api_url': 'https://developer.uspto.gov/api',
                'formats': ['xml', 'json']
            },
            'epo': {
                'ops_url': 'https://ops.epo.org/3.2',
                'espacenet_url': 'https://worldwide.espacenet.com',
                'formats': ['xml', 'json']
            },
            'kipo': {
                'kipris_url': 'https://www.kipris.or.kr',
                'formats': ['xml', 'html']
            }
        }
    
    def collect_uspto_bulk_data(self, year: int, data_type: str = 'grant') -> str:
        """USPTO 대량 데이터 수집"""
        try:
            # 대량 데이터 다운로드 URL 구성
            if data_type == 'grant':
                url = f"https://data.uspto.gov/bulkdata/patent/grant/redbook/fulltext/{year}/"
            else:
                url = f"https://data.uspto.gov/bulkdata/patent/application/redbook/fulltext/{year}/"
            
            # 데이터 다운로드 디렉토리 생성
            download_dir = self.data_dir / f"uspto_{data_type}_{year}"
            download_dir.mkdir(exist_ok=True)
            
            logger.info(f"USPTO {data_type} 데이터 수집 시작: {year}년")
            
            # 실제 구현에서는 requests를 사용하여 파일 다운로드
            # 여기서는 예시 구조만 제공
            
            return str(download_dir)
            
        except Exception as e:
            logger.error(f"USPTO 데이터 수집 오류: {e}")
            return ""
    
    def parse_uspto_xml(self, xml_file_path: str) -> List[Dict]:
        """USPTO XML 파일 파싱"""
        patents = []
        
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # XML 구조에 따라 특허 정보 추출
            for patent_elem in root.findall('.//us-patent-grant'):
                patent_data = self._extract_patent_from_xml(patent_elem)
                if patent_data:
                    patents.append(patent_data)
            
        except Exception as e:
            logger.error(f"XML 파싱 오류: {e}")
        
        return patents
    
    def _extract_patent_from_xml(self, patent_elem: ET.Element) -> Optional[Dict]:
        """XML 요소에서 특허 정보 추출"""
        try:
            # 기본 정보 추출
            patent_id = patent_elem.find('.//doc-number')
            title = patent_elem.find('.//invention-title')
            abstract = patent_elem.find('.//abstract')
            
            # 클레임 추출
            claims = []
            for claim_elem in patent_elem.findall('.//claim'):
                claim_text = ET.tostring(claim_elem, encoding='unicode', method='text')
                claims.append(claim_text.strip())
            
            # 발명자 정보 추출
            inventors = []
            for inventor_elem in patent_elem.findall('.//inventor'):
                name_elem = inventor_elem.find('.//name')
                if name_elem is not None:
                    inventors.append(ET.tostring(name_elem, encoding='unicode', method='text').strip())
            
            patent_data = {
                'patent_id': patent_id.text if patent_id is not None else '',
                'title': title.text if title is not None else '',
                'abstract': ET.tostring(abstract, encoding='unicode', method='text').strip() if abstract is not None else '',
                'claims': claims,
                'inventors': inventors,
                'source': 'uspto'
            }
            
            return patent_data
            
        except Exception as e:
            logger.error(f"특허 정보 추출 오류: {e}")
            return None

class PatentDataPreprocessor:
    """특허 데이터 전처리기 통합 클래스"""
    
    def __init__(self, output_db_path: str = "processed_patents.db"):
        self.text_processor = PatentTextProcessor()
        self.figure_processor = PatentFigureProcessor()
        self.data_collector = PatentDataCollector()
        
        self.output_db_path = output_db_path
        self.init_database()
    
    def init_database(self):
        """전처리 결과 데이터베이스 초기화"""
        conn = sqlite3.connect(self.output_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_patents (
                patent_id TEXT PRIMARY KEY,
                processed_text TEXT,
                structured_claims TEXT,
                extracted_figures TEXT,
                technical_keywords TEXT,
                classification_hierarchy TEXT,
                processing_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_patent_document(self, patent_data: Dict, figure_dir: Optional[str] = None) -> ProcessedPatentData:
        """특허 문서 전체 처리"""
        
        logger.info(f"특허 문서 처리 시작: {patent_data.get('patent_id', 'Unknown')}")
        
        # 텍스트 전처리
        raw_text = f"{patent_data.get('title', '')} {patent_data.get('abstract', '')} {' '.join(patent_data.get('claims', []))}"
        processed_text = self.text_processor.clean_patent_text(raw_text)
        
        # 클레임 구조화
        structured_claims = self.text_processor.extract_claims(raw_text)
        
        # 기술 키워드 추출
        technical_keywords = self.text_processor.extract_technical_keywords(processed_text)
        
        # 도면 처리
        extracted_figures = []
        if figure_dir and os.path.exists(figure_dir):
            for figure_file in os.listdir(figure_dir):
                if any(figure_file.lower().endswith(ext) for ext in self.figure_processor.supported_formats):
                    figure_path = os.path.join(figure_dir, figure_file)
                    figure_analysis = self.figure_processor.analyze_figure_content(figure_path)
                    extracted_figures.append(figure_analysis)
        
        # 분류 계층 구조 생성
        classification_hierarchy = self._build_classification_hierarchy(patent_data, technical_keywords)
        
        # 처리 메타데이터
        processing_metadata = {
            'processing_date': datetime.now().isoformat(),
            'text_length': len(processed_text),
            'claim_count': len(structured_claims),
            'figure_count': len(extracted_figures),
            'keyword_count': len(technical_keywords)
        }
        
        processed_data = ProcessedPatentData(
            patent_id=patent_data.get('patent_id', ''),
            processed_text=processed_text,
            structured_claims=structured_claims,
            extracted_figures=extracted_figures,
            technical_keywords=technical_keywords,
            classification_hierarchy=classification_hierarchy,
            processing_metadata=processing_metadata
        )
        
        # 데이터베이스에 저장
        self._save_processed_data(processed_data)
        
        logger.info(f"특허 문서 처리 완료: {processed_data.patent_id}")
        return processed_data
    
    def _build_classification_hierarchy(self, patent_data: Dict, keywords: List[str]) -> Dict:
        """분류 계층 구조 구축"""
        hierarchy = {
            'primary_category': 'unknown',
            'secondary_categories': [],
            'technical_domains': [],
            'application_areas': []
        }
        
        # 키워드 기반 분류
        keyword_categories = {
            'ai_ml': ['artificial intelligence', 'machine learning', 'neural network'],
            'electronics': ['circuit', 'semiconductor', 'electronic'],
            'mechanical': ['mechanical', 'motor', 'gear'],
            'software': ['software', 'algorithm', 'program'],
            'biotech': ['biological', 'medical', 'pharmaceutical']
        }
        
        for category, category_keywords in keyword_categories.items():
            if any(kw in ' '.join(keywords).lower() for kw in category_keywords):
                if hierarchy['primary_category'] == 'unknown':
                    hierarchy['primary_category'] = category
                else:
                    hierarchy['secondary_categories'].append(category)
        
        return hierarchy
    
    def _save_processed_data(self, processed_data: ProcessedPatentData):
        """처리된 데이터 저장"""
        conn = sqlite3.connect(self.output_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO processed_patents 
            (patent_id, processed_text, structured_claims, extracted_figures,
             technical_keywords, classification_hierarchy, processing_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            processed_data.patent_id,
            processed_data.processed_text,
            json.dumps(processed_data.structured_claims),
            json.dumps(processed_data.extracted_figures),
            json.dumps(processed_data.technical_keywords),
            json.dumps(processed_data.classification_hierarchy),
            json.dumps(processed_data.processing_metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def batch_process_patents(self, patent_data_list: List[Dict]) -> List[ProcessedPatentData]:
        """특허 데이터 배치 처리"""
        processed_patents = []
        
        for i, patent_data in enumerate(patent_data_list):
            try:
                logger.info(f"배치 처리 진행: {i+1}/{len(patent_data_list)}")
                processed_patent = self.process_patent_document(patent_data)
                processed_patents.append(processed_patent)
                
            except Exception as e:
                logger.error(f"특허 처리 오류 ({patent_data.get('patent_id', 'Unknown')}): {e}")
        
        logger.info(f"배치 처리 완료: {len(processed_patents)}개 특허 처리됨")
        return processed_patents
    
    def get_processing_statistics(self) -> Dict:
        """처리 통계 조회"""
        conn = sqlite3.connect(self.output_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM processed_patents")
        total_patents = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(json_extract(processing_metadata, '$.text_length')) FROM processed_patents")
        avg_text_length = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT AVG(json_extract(processing_metadata, '$.claim_count')) FROM processed_patents")
        avg_claim_count = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_processed_patents': total_patents,
            'average_text_length': round(avg_text_length, 2),
            'average_claim_count': round(avg_claim_count, 2)
        }

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🔧 특허 데이터 처리 및 전처리 시스템")
    print("Patent Data Processing and Preprocessing System")
    print("=" * 80)
    
    # 전처리기 초기화
    preprocessor = PatentDataPreprocessor()
    
    while True:
        print("\n📋 메뉴를 선택하세요:")
        print("1. 📄 단일 특허 문서 처리")
        print("2. 📦 배치 특허 데이터 처리")
        print("3. 📊 처리 통계 조회")
        print("4. 🔍 USPTO 데이터 수집")
        print("5. ❌ 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == '1':
            # 샘플 특허 데이터로 테스트
            sample_patent = {
                'patent_id': 'US1234567A',
                'title': 'Artificial Intelligence System for Patent Analysis',
                'abstract': 'This invention relates to an artificial intelligence system that analyzes patent documents using machine learning algorithms.',
                'claims': [
                    'A method for analyzing patent documents comprising: receiving patent text, processing with neural networks, generating analysis results.',
                    'The method of claim 1, wherein the neural network is a transformer-based language model.'
                ]
            }
            
            processed = preprocessor.process_patent_document(sample_patent)
            print(f"\n✅ 특허 처리 완료: {processed.patent_id}")
            print(f"   - 처리된 텍스트 길이: {len(processed.processed_text)}")
            print(f"   - 구조화된 클레임: {len(processed.structured_claims)}개")
            print(f"   - 기술 키워드: {len(processed.technical_keywords)}개")
        
        elif choice == '2':
            print("배치 처리 기능은 실제 특허 데이터가 필요합니다.")
            
        elif choice == '3':
            stats = preprocessor.get_processing_statistics()
            print(f"\n📊 처리 통계:")
            print(f"- 총 처리된 특허: {stats['total_processed_patents']}개")
            print(f"- 평균 텍스트 길이: {stats['average_text_length']}")
            print(f"- 평균 클레임 수: {stats['average_claim_count']}")
        
        elif choice == '4':
            year = input("수집할 연도를 입력하세요 (예: 2023): ").strip()
            if year.isdigit():
                collector = PatentDataCollector()
                result_dir = collector.collect_uspto_bulk_data(int(year))
                print(f"✅ 데이터 수집 디렉토리: {result_dir}")
            else:
                print("❌ 올바른 연도를 입력해주세요.")
        
        elif choice == '5':
            print("시스템을 종료합니다.")
            break
        
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()

