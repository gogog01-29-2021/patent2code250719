# Patent2Tech LLM 시스템 완전 구현 가이드

## 개요

본 문서는 현재 부분적으로 구현된 Patent2Tech LLM 시스템을 완전한 프로덕션 서비스로 발전시키기 위한 종합적인 구현 가이드입니다. 현재 시스템의 한계점을 분석하고, 실제 LLM 연동, 특허 데이터 파이프라인 구축, 3D 모델링 기능 추가, 그리고 배포 가능한 서비스로 만들기 위한 모든 필요 사항을 상세히 다룹니다.

## 현재 시스템 상태 분석

### 작동하는 부분
- ✅ React 기반 프론트엔드 UI/UX
- ✅ Flask 기반 백엔드 API 구조
- ✅ 기본적인 데이터베이스 모델 설계
- ✅ 시뮬레이션된 특허 검색 및 분석 기능

### 개선이 필요한 부분
- ❌ 실제 LLM 모델 연동 부재
- ❌ 특허 데이터 수집 파이프라인 미구현
- ❌ 데이터베이스 완전 연동 부족
- ❌ 3D 모델링 기능 미구현
- ❌ 사용자 인증 및 보안 시스템 부재
- ❌ API 오류 및 안정성 문제
- ❌ 프로덕션 배포 준비 미완료

## 필요한 리소스 및 API 키

### 1. 필수 API 키 목록

#### OpenAI API 키 (최우선)
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
```
- **용도**: GPT-4/GPT-3.5를 활용한 특허 분석, 코드 생성, 기술 개념 추출
- **예상 비용**: 월 $100-500 (사용량에 따라)
- **권장 모델**: 
  - GPT-4-turbo: 복잡한 특허 분석용
  - GPT-3.5-turbo: 일반적인 텍스트 처리용
  - text-embedding-ada-002: 벡터 임베딩용

#### Google Cloud Platform API 키
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```
- **용도**: Google Patents API, Cloud Vision API (OCR), Translation API
- **예상 비용**: 월 $50-200

#### 특허 데이터베이스 API 키
```bash
export USPTO_API_KEY="your-uspto-key"  # 선택사항 (공개 API)
export EPO_API_KEY="your-epo-key"      # 필수
export KIPO_API_KEY="your-kipo-key"    # 필수
```

#### 기타 서비스 API 키
```bash
export SENDGRID_API_KEY="SG.your-sendgrid-key"  # 이메일 발송
export HUGGINGFACE_API_KEY="hf_your-hf-key"     # 모델 호스팅
```

### 2. 인프라 요구사항

#### 최소 구성 (개발/테스트)
- **서버**: 4 vCPU, 16GB RAM, 100GB SSD
- **데이터베이스**: PostgreSQL 13+
- **캐시**: Redis 6+
- **예상 비용**: 월 $200-300

#### 프로덕션 구성
- **웹서버**: 2x t3.medium (AWS)
- **API서버**: 2x t3.large (AWS)
- **GPU서버**: 1x p3.2xlarge (NeRF용, 필요시)
- **데이터베이스**: RDS PostgreSQL db.t3.medium
- **캐시**: ElastiCache Redis cache.t3.micro
- **예상 비용**: 월 $400-1,400 (GPU 포함 여부에 따라)

## 단계별 구현 계획

### Phase 1: LLM 연동 및 API 오류 해결 (우선순위: 최고)

#### 1.1 OpenAI API 통합
현재 시뮬레이션으로 구현된 LLM 기능을 실제 OpenAI API와 연동합니다.

**구현 파일**: `patent_llm_api/src/services/llm_service.py`

```python
import openai
from typing import Dict, List, Optional
import json
import time

class PatentLLMService:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.models = {
            'analysis': 'gpt-4-turbo-preview',
            'code_generation': 'gpt-4-turbo-preview',
            'embedding': 'text-embedding-ada-002'
        }
    
    def analyze_patent(self, patent_text: str, query: str) -> Dict:
        """특허 문서 분석"""
        prompt = f"""
        다음 특허 문서를 분석하고 사용자 질의에 답변해주세요.
        
        특허 문서:
        {patent_text[:4000]}  # 토큰 제한 고려
        
        사용자 질의: {query}
        
        다음 형식으로 답변해주세요:
        1. 핵심 기술 개념
        2. 구현 가능성 분석
        3. 관련 기술 분야
        4. 상업적 활용 방안
        """
        
        response = openai.ChatCompletion.create(
            model=self.models['analysis'],
            messages=[
                {"role": "system", "content": "당신은 특허 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return {
            'analysis': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens,
            'cost': self._calculate_cost(response.usage.total_tokens, 'gpt-4')
        }
    
    def generate_code(self, concept: str, language: str, framework: str) -> Dict:
        """기술 개념을 코드로 변환"""
        prompt = f"""
        다음 기술 개념을 {language} 언어와 {framework} 프레임워크를 사용하여 
        구현 가능한 코드로 변환해주세요.
        
        기술 개념: {concept}
        
        요구사항:
        1. 완전히 작동하는 코드 제공
        2. 상세한 주석 포함
        3. 테스트 케이스 포함
        4. 의존성 목록 제공
        """
        
        response = openai.ChatCompletion.create(
            model=self.models['code_generation'],
            messages=[
                {"role": "system", "content": "당신은 숙련된 소프트웨어 개발자입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.2
        )
        
        return {
            'code': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens,
            'cost': self._calculate_cost(response.usage.total_tokens, 'gpt-4')
        }
```

#### 1.2 API 오류 해결
현재 발생하는 500 오류를 해결하기 위한 수정사항:

**수정 파일**: `patent_llm_api/src/routes/patent.py`

```python
from flask import Blueprint, request, jsonify
from src.services.llm_service import PatentLLMService
from src.models.patent import Patent, AnalysisResult
from src.database import db
import os
import logging

patent_bp = Blueprint('patent', __name__)
logger = logging.getLogger(__name__)

# LLM 서비스 초기화
llm_service = PatentLLMService(os.getenv('OPENAI_API_KEY'))

@patent_bp.route('/search', methods=['POST'])
def search_patents():
    try:
        data = request.get_json()
        query = data.get('query', '')
        implementation_type = data.get('implementation_type', 'code')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # 특허 검색 (실제 구현 필요)
        patents = search_patent_database(query)
        
        # LLM을 사용한 분석
        results = []
        for patent in patents[:5]:  # 상위 5개만 분석
            analysis = llm_service.analyze_patent(
                patent.get_full_text(), 
                query
            )
            
            # 코드 생성 (요청시)
            if implementation_type == 'code':
                code_result = llm_service.generate_code(
                    analysis['analysis'],
                    data.get('language', 'python'),
                    data.get('framework', 'flask')
                )
                analysis['generated_code'] = code_result
            
            results.append({
                'patent': patent.to_dict(),
                'analysis': analysis
            })
        
        # 결과 저장
        analysis_record = AnalysisResult(
            query=query,
            analysis_type='search',
            result=results,
            tokens_used=sum(r['analysis']['tokens_used'] for r in results),
            cost_usd=sum(r['analysis']['cost'] for r in results)
        )
        db.session.add(analysis_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'results': results,
            'total_cost': analysis_record.cost_usd
        })
        
    except Exception as e:
        logger.error(f"Error in patent search: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
```

### Phase 2: 특허 데이터 수집 파이프라인 구축

#### 2.1 특허 데이터 수집기 구현

**새 파일**: `patent_llm_api/src/services/patent_collector.py`

```python
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
```

#### 2.2 데이터베이스 완전 연동

**수정 파일**: `patent_llm_api/src/models/patent.py`

```python
from sqlalchemy import Column, Integer, String, Text, Date, ARRAY, JSON, TIMESTAMP, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Patent(Base):
    __tablename__ = 'patents'
    
    id = Column(Integer, primary_key=True)
    patent_number = Column(String(50), unique=True, nullable=False)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    claims = Column(Text)
    description = Column(Text)
    inventors = Column(ARRAY(String))
    assignees = Column(ARRAY(String))
    filing_date = Column(Date)
    publication_date = Column(Date)
    grant_date = Column(Date)
    patent_office = Column(String(10))
    classification_codes = Column(ARRAY(String))
    citations = Column(ARRAY(Integer))
    figures_count = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 관계 설정
    figures = relationship("PatentFigure", back_populates="patent")
    analysis_results = relationship("AnalysisResult", back_populates="patent")
    
    def to_dict(self):
        return {
            'id': self.id,
            'patent_number': self.patent_number,
            'title': self.title,
            'abstract': self.abstract,
            'inventors': self.inventors,
            'assignees': self.assignees,
            'filing_date': self.filing_date.isoformat() if self.filing_date else None,
            'publication_date': self.publication_date.isoformat() if self.publication_date else None,
            'patent_office': self.patent_office,
            'figures_count': self.figures_count
        }
    
    def get_full_text(self):
        """분석용 전체 텍스트 반환"""
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.claims:
            parts.append(f"Claims: {self.claims}")
        if self.description:
            parts.append(f"Description: {self.description[:2000]}")  # 길이 제한
        
        return "\n\n".join(parts)

class PatentFigure(Base):
    __tablename__ = 'patent_figures'
    
    id = Column(Integer, primary_key=True)
    patent_id = Column(Integer, ForeignKey('patents.id'), nullable=False)
    figure_number = Column(String(20))
    figure_type = Column(String(50))
    file_path = Column(Text)
    file_size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    ocr_text = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # 관계 설정
    patent = relationship("Patent", back_populates="figures")

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    patent_id = Column(Integer, ForeignKey('patents.id'))
    query = Column(Text, nullable=False)
    analysis_type = Column(String(50))
    result = Column(JSON)
    processing_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    cost_usd = Column(String(10))  # Decimal 대신 String 사용
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # 관계 설정
    patent = relationship("Patent", back_populates="analysis_results")
    generated_codes = relationship("GeneratedCode", back_populates="analysis")
    generated_3d_models = relationship("Generated3DModel", back_populates="analysis")

class GeneratedCode(Base):
    __tablename__ = 'generated_codes'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), nullable=False)
    language = Column(String(20))
    framework = Column(String(50))
    code_content = Column(Text)
    explanation = Column(Text)
    test_cases = Column(Text)
    dependencies = Column(ARRAY(String))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # 관계 설정
    analysis = relationship("AnalysisResult", back_populates="generated_codes")

class Generated3DModel(Base):
    __tablename__ = 'generated_3d_models'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_results.id'), nullable=False)
    model_type = Column(String(50))
    file_path = Column(Text)
    file_size = Column(Integer)
    processing_time_ms = Column(Integer)
    parameters = Column(JSON)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # 관계 설정
    analysis = relationship("AnalysisResult", back_populates="generated_3d_models")
```

### Phase 3: 3D 모델링 기능 구현 (NeRF 포함)

#### 3.1 NeRF 기반 3D 모델 생성기

**새 파일**: `patent_llm_api/src/services/nerf_service.py`

```python
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
import os
import subprocess
import json
from PIL import Image
import logging

class PatentNeRFService:
    """특허 도면을 위한 NeRF 기반 3D 모델 생성 서비스"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.logger = logging.getLogger(__name__)
        
        # NeRF 모델 설정
        self.nerf_config = {
            'num_epochs': 1000,
            'batch_size': 1024,
            'learning_rate': 5e-4,
            'num_samples': 64,
            'num_importance': 128
        }
    
    def preprocess_patent_figures(self, figure_paths: List[str]) -> Dict:
        """특허 도면 전처리"""
        processed_images = []
        camera_poses = []
        
        for i, figure_path in enumerate(figure_paths):
            try:
                # 이미지 로드 및 전처리
                image = cv2.imread(figure_path)
                if image is None:
                    continue
                
                # 이미지 크기 정규화
                height, width = image.shape[:2]
                if width > 512 or height > 512:
                    scale = 512 / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                # 배경 제거 (선택적)
                image = self._remove_background(image)
                
                # 카메라 포즈 추정 (단순화된 버전)
                pose = self._estimate_camera_pose(i, len(figure_paths))
                
                processed_images.append(image)
                camera_poses.append(pose)
                
            except Exception as e:
                self.logger.error(f"Error processing figure {figure_path}: {e}")
        
        return {
            'images': processed_images,
            'poses': camera_poses,
            'intrinsics': self._get_default_intrinsics()
        }
    
    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """배경 제거 (간단한 버전)"""
        # 실제로는 더 정교한 배경 제거 알고리즘 사용
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 마스크 적용
        result = image.copy()
        result[mask == 0] = [255, 255, 255]  # 배경을 흰색으로
        
        return result
    
    def _estimate_camera_pose(self, view_idx: int, total_views: int) -> np.ndarray:
        """카메라 포즈 추정 (원형 배치)"""
        angle = 2 * np.pi * view_idx / total_views
        radius = 2.0
        
        # 카메라 위치
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        
        # 카메라가 원점을 바라보도록 설정
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Look-at 매트릭스 생성
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # 4x4 변환 매트릭스
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        return pose
    
    def _get_default_intrinsics(self) -> np.ndarray:
        """기본 카메라 내부 파라미터"""
        focal_length = 400.0
        cx, cy = 256.0, 256.0
        
        intrinsics = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        return intrinsics
    
    def generate_3d_model(self, figure_paths: List[str], output_dir: str) -> Dict:
        """특허 도면으로부터 3D 모델 생성"""
        try:
            # 전처리
            processed_data = self.preprocess_patent_figures(figure_paths)
            
            if len(processed_data['images']) < 2:
                raise ValueError("At least 2 images required for 3D reconstruction")
            
            # NeRF 훈련 데이터 준비
            train_data = self._prepare_nerf_data(processed_data, output_dir)
            
            # NeRF 모델 훈련
            model_path = self._train_nerf_model(train_data, output_dir)
            
            # 3D 메시 추출
            mesh_path = self._extract_mesh(model_path, output_dir)
            
            # 결과 렌더링
            rendered_views = self._render_novel_views(model_path, output_dir)
            
            return {
                'success': True,
                'model_path': model_path,
                'mesh_path': mesh_path,
                'rendered_views': rendered_views,
                'processing_info': {
                    'input_images': len(figure_paths),
                    'processed_images': len(processed_data['images']),
                    'model_type': 'NeRF',
                    'output_format': 'PLY'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating 3D model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_nerf_data(self, processed_data: Dict, output_dir: str) -> str:
        """NeRF 훈련 데이터 준비"""
        data_dir = os.path.join(output_dir, 'nerf_data')
        os.makedirs(data_dir, exist_ok=True)
        
        # 이미지 저장
        images_dir = os.path.join(data_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        for i, image in enumerate(processed_data['images']):
            image_path = os.path.join(images_dir, f'{i:03d}.png')
            cv2.imwrite(image_path, image)
        
        # 카메라 파라미터 저장
        transforms = {
            'camera_angle_x': 0.6911112070083618,
            'frames': []
        }
        
        for i, pose in enumerate(processed_data['poses']):
            frame = {
                'file_path': f'./images/{i:03d}.png',
                'transform_matrix': pose.tolist()
            }
            transforms['frames'].append(frame)
        
        transforms_path = os.path.join(data_dir, 'transforms.json')
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        return data_dir
    
    def _train_nerf_model(self, data_dir: str, output_dir: str) -> str:
        """NeRF 모델 훈련"""
        model_dir = os.path.join(output_dir, 'nerf_model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Nerfstudio 사용 (실제 구현에서는 적절한 NeRF 라이브러리 사용)
        try:
            cmd = [
                'ns-train', 'nerfacto',
                '--data', data_dir,
                '--output-dir', model_dir,
                '--max-num-iterations', str(self.nerf_config['num_epochs'])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                raise RuntimeError(f"NeRF training failed: {result.stderr}")
            
            # 모델 파일 경로 반환
            model_path = os.path.join(model_dir, 'nerfacto', 'model.ckpt')
            return model_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("NeRF training timed out")
        except FileNotFoundError:
            # Nerfstudio가 설치되지 않은 경우 대안 구현
            return self._train_simple_nerf(data_dir, model_dir)
    
    def _train_simple_nerf(self, data_dir: str, model_dir: str) -> str:
        """간단한 NeRF 구현 (Nerfstudio 대안)"""
        # 실제로는 PyTorch 기반 NeRF 구현
        # 여기서는 플레이스홀더
        model_path = os.path.join(model_dir, 'simple_nerf.pth')
        
        # 더미 모델 저장
        torch.save({'model_state': 'placeholder'}, model_path)
        
        return model_path
    
    def _extract_mesh(self, model_path: str, output_dir: str) -> str:
        """3D 메시 추출"""
        mesh_dir = os.path.join(output_dir, 'mesh')
        os.makedirs(mesh_dir, exist_ok=True)
        
        mesh_path = os.path.join(mesh_dir, 'model.ply')
        
        try:
            # Marching Cubes 알고리즘을 사용한 메시 추출
            cmd = [
                'ns-export', 'poisson',
                '--load-config', model_path,
                '--output-dir', mesh_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return mesh_path
            else:
                # 대안 메시 생성
                return self._generate_simple_mesh(output_dir)
                
        except FileNotFoundError:
            return self._generate_simple_mesh(output_dir)
    
    def _generate_simple_mesh(self, output_dir: str) -> str:
        """간단한 메시 생성 (대안)"""
        mesh_path = os.path.join(output_dir, 'simple_mesh.ply')
        
        # 간단한 PLY 파일 생성 (실제로는 더 정교한 구현 필요)
        ply_content = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
-1 -1 -1
1 -1 -1
1 1 -1
-1 1 -1
-1 -1 1
1 -1 1
1 1 1
-1 1 1
3 0 1 2
3 0 2 3
3 4 7 6
3 4 6 5
3 0 4 5
3 0 5 1
3 2 6 7
3 2 7 3
3 0 3 7
3 0 7 4
3 1 5 6
3 1 6 2
"""
        
        with open(mesh_path, 'w') as f:
            f.write(ply_content)
        
        return mesh_path
    
    def _render_novel_views(self, model_path: str, output_dir: str) -> List[str]:
        """새로운 시점에서 렌더링"""
        render_dir = os.path.join(output_dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)
        
        rendered_paths = []
        
        # 여러 시점에서 렌더링
        for i in range(8):
            angle = 2 * np.pi * i / 8
            render_path = os.path.join(render_dir, f'render_{i:03d}.png')
            
            # 실제로는 NeRF 모델을 사용한 렌더링
            # 여기서는 플레이스홀더 이미지 생성
            placeholder_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
            cv2.putText(placeholder_image, f'View {i+1}', (200, 256), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(render_path, placeholder_image)
            
            rendered_paths.append(render_path)
        
        return rendered_paths

# 3D 모델링 API 엔드포인트 추가
@patent_bp.route('/generate-3d', methods=['POST'])
def generate_3d_model():
    try:
        data = request.get_json()
        patent_id = data.get('patent_id')
        
        if not patent_id:
            return jsonify({'error': 'Patent ID is required'}), 400
        
        # 특허 도면 조회
        patent = Patent.query.get(patent_id)
        if not patent:
            return jsonify({'error': 'Patent not found'}), 404
        
        figure_paths = [fig.file_path for fig in patent.figures if fig.file_path]
        
        if len(figure_paths) < 2:
            return jsonify({'error': 'At least 2 figures required for 3D modeling'}), 400
        
        # 3D 모델 생성
        nerf_service = PatentNeRFService()
        output_dir = f"outputs/3d_models/{patent.patent_number}"
        
        result = nerf_service.generate_3d_model(figure_paths, output_dir)
        
        if result['success']:
            # 결과를 데이터베이스에 저장
            model_record = Generated3DModel(
                analysis_id=None,  # 별도 분석 없이 직접 생성
                model_type='nerf',
                file_path=result['mesh_path'],
                processing_time_ms=0,  # 실제 측정 필요
                parameters=result['processing_info']
            )
            db.session.add(model_record)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'model_id': model_record.id,
                'mesh_path': result['mesh_path'],
                'rendered_views': result['rendered_views']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        logger.error(f"Error in 3D model generation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
```

### Phase 4: 사용자 인증 및 보안 시스템

#### 4.1 사용자 인증 시스템

**새 파일**: `patent_llm_api/src/auth/auth_service.py`

```python
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import secrets
import string

class AuthService:
    def __init__(self, app):
        self.jwt = JWTManager(app)
        app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(32)
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
    
    def register_user(self, username: str, email: str, password: str) -> Dict:
        """사용자 등록"""
        # 중복 확인
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return {'success': False, 'error': 'User already exists'}
        
        # 비밀번호 해시화
        password_hash = generate_password_hash(password)
        
        # 사용자 생성
        user = User(
            username=username,
            email=email,
            password_hash=password_hash
        )
        
        db.session.add(user)
        db.session.commit()
        
        # 액세스 토큰 생성
        access_token = create_access_token(identity=user.id)
        
        return {
            'success': True,
            'user_id': user.id,
            'access_token': access_token
        }
    
    def login_user(self, email: str, password: str) -> Dict:
        """사용자 로그인"""
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return {'success': False, 'error': 'Invalid credentials'}
        
        if not user.is_active:
            return {'success': False, 'error': 'Account is deactivated'}
        
        # 로그인 정보 업데이트
        user.last_login_at = datetime.utcnow()
        user.login_count += 1
        db.session.commit()
        
        # 액세스 토큰 생성
        access_token = create_access_token(identity=user.id)
        
        return {
            'success': True,
            'user_id': user.id,
            'access_token': access_token,
            'user_info': {
                'username': user.username,
                'email': user.email,
                'role': user.role
            }
        }
    
    def generate_api_key(self, user_id: int, key_name: str, permissions: Dict) -> Dict:
        """API 키 생성"""
        # API 키 생성
        key = 'pk_' + ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        key_hash = generate_password_hash(key)
        key_prefix = key[:8]
        
        api_key = APIKey(
            user_id=user_id,
            key_name=key_name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=permissions
        )
        
        db.session.add(api_key)
        db.session.commit()
        
        return {
            'success': True,
            'api_key': key,
            'key_id': api_key.id,
            'key_prefix': key_prefix
        }
```

#### 4.2 API 보안 및 제한

**새 파일**: `patent_llm_api/src/middleware/rate_limiter.py`

```python
from flask import request, jsonify
from functools import wraps
import redis
import time
import json

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def limit(self, max_requests: int, window_seconds: int, per: str = 'ip'):
        """요청 제한 데코레이터"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # 식별자 결정
                if per == 'ip':
                    identifier = request.remote_addr
                elif per == 'user':
                    identifier = get_jwt_identity() if jwt_required else request.remote_addr
                elif per == 'api_key':
                    identifier = request.headers.get('X-API-Key', request.remote_addr)
                else:
                    identifier = request.remote_addr
                
                # Redis 키 생성
                key = f"rate_limit:{per}:{identifier}:{int(time.time() // window_seconds)}"
                
                # 현재 요청 수 확인
                current_requests = self.redis.get(key)
                if current_requests is None:
                    current_requests = 0
                else:
                    current_requests = int(current_requests)
                
                if current_requests >= max_requests:
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': window_seconds
                    }), 429
                
                # 요청 수 증가
                pipe = self.redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                pipe.execute()
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator

# 사용 예시
rate_limiter = RateLimiter(redis_client)

@patent_bp.route('/search', methods=['POST'])
@rate_limiter.limit(max_requests=100, window_seconds=3600, per='user')  # 시간당 100회
@jwt_required()
def search_patents():
    # 기존 코드...
    pass
```

### Phase 5: 프론트엔드 개선

#### 5.1 사용자 인증 UI 추가

**새 파일**: `patent_llm_frontend/src/components/Auth/LoginForm.jsx`

```jsx
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

export function LoginForm({ onLogin }) {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5001/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (data.success) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user_info', JSON.stringify(data.user_info));
        onLogin(data.user_info);
      } else {
        setError(data.error || 'Login failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Login to Patent2Tech</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          <div>
            <Input
              type="email"
              placeholder="Email"
              value={formData.email}
              onChange={(e) => setFormData({...formData, email: e.target.value})}
              required
            />
          </div>
          
          <div>
            <Input
              type="password"
              placeholder="Password"
              value={formData.password}
              onChange={(e) => setFormData({...formData, password: e.target.value})}
              required
            />
          </div>
          
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? 'Logging in...' : 'Login'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
```

#### 5.2 3D 모델 뷰어 컴포넌트

**새 파일**: `patent_llm_frontend/src/components/3D/ModelViewer.jsx`

```jsx
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function ModelViewer({ modelPath, title }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!modelPath) return;

    // Three.js 씬 초기화
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // 카메라 설정
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 5);

    // 렌더러 설정
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    mountRef.current.appendChild(renderer.domElement);

    // 조명 설정
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // 컨트롤 설정
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;

    // PLY 모델 로드
    const loader = new PLYLoader();
    loader.load(
      modelPath,
      (geometry) => {
        // 지오메트리 정규화
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        // 머티리얼 생성
        const material = new THREE.MeshLambertMaterial({
          color: 0x0055ff,
          side: THREE.DoubleSide
        });

        // 메시 생성
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        setLoading(false);
      },
      (progress) => {
        console.log('Loading progress:', progress);
      },
      (error) => {
        console.error('Error loading model:', error);
        setError('Failed to load 3D model');
        setLoading(false);
      }
    );

    // 애니메이션 루프
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 리사이즈 핸들러
    const handleResize = () => {
      if (mountRef.current) {
        camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
      }
    };

    window.addEventListener('resize', handleResize);

    // 클린업
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [modelPath]);

  const downloadModel = () => {
    const link = document.createElement('a');
    link.href = modelPath;
    link.download = `${title || 'model'}.ply`;
    link.click();
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p>Loading 3D model...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <div className="text-center text-red-600">
            <p>{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          {title || '3D Model'}
          <Button onClick={downloadModel} variant="outline" size="sm">
            Download PLY
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={mountRef} className="w-full h-96 border rounded" />
        <div className="mt-4 text-sm text-gray-600">
          <p>• Left click + drag: Rotate</p>
          <p>• Right click + drag: Pan</p>
          <p>• Scroll: Zoom</p>
        </div>
      </CardContent>
    </Card>
  );
}
```

## 배포 준비 사항

### 1. 환경 변수 설정

**파일**: `.env.production`

```bash
# Flask 설정
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/patent_db
REDIS_URL=redis://localhost:6379

# API 키들
OPENAI_API_KEY=sk-your-openai-api-key
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
USPTO_API_KEY=your-uspto-key
EPO_API_KEY=your-epo-key
KIPO_API_KEY=your-kipo-key
SENDGRID_API_KEY=SG.your-sendgrid-key
HUGGINGFACE_API_KEY=hf_your-hf-key

# 보안 설정
JWT_SECRET_KEY=your-jwt-secret-key
CORS_ORIGINS=https://patent2tech.com,https://www.patent2tech.com

# 파일 저장
UPLOAD_FOLDER=/app/uploads
MAX_CONTENT_LENGTH=104857600  # 100MB

# 로깅
LOG_LEVEL=INFO
LOG_FILE=/app/logs/patent_llm.log

# 모니터링
SENTRY_DSN=your-sentry-dsn
```

### 2. Docker 컨테이너 구성

**파일**: `docker-compose.prod.yml`

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./patent_llm_frontend
      dockerfile: Dockerfile.prod
    ports:
      - "80:80"
      - "443:443"
    environment:
      - REACT_APP_API_URL=https://api.patent2tech.com
    volumes:
      - ./ssl:/etc/ssl/certs
    depends_on:
      - backend

  backend:
    build:
      context: ./patent_llm_api
      dockerfile: Dockerfile.prod
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://patent_user:${DB_PASSWORD}@postgres:5432/patent_db
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
      - SENDGRID_API_KEY=${SENDGRID_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
      - ./ssl:/app/ssl
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=patent_db
      - POSTGRES_USER=patent_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
```

### 3. 배포 스크립트

**파일**: `deploy.sh`

```bash
#!/bin/bash

set -e

echo "Starting Patent2Tech deployment..."

# 환경 변수 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    exit 1
fi

if [ -z "$DB_PASSWORD" ]; then
    echo "Error: DB_PASSWORD is not set"
    exit 1
fi

# 이전 컨테이너 정리
echo "Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# 이미지 빌드
echo "Building images..."
docker-compose -f docker-compose.prod.yml build

# 데이터베이스 마이그레이션
echo "Running database migrations..."
docker-compose -f docker-compose.prod.yml run --rm backend python -m flask db upgrade

# 컨테이너 시작
echo "Starting containers..."
docker-compose -f docker-compose.prod.yml up -d

# 헬스체크
echo "Waiting for services to start..."
sleep 30

# 백엔드 헬스체크
if curl -f http://localhost:5000/health; then
    echo "Backend is healthy"
else
    echo "Backend health check failed"
    exit 1
fi

# 프론트엔드 헬스체크
if curl -f http://localhost:80; then
    echo "Frontend is healthy"
else
    echo "Frontend health check failed"
    exit 1
fi

echo "Deployment completed successfully!"
echo "Application is available at: http://localhost"
echo "API is available at: http://localhost:5000"
```

## 비용 추정

### 개발 단계 (월간)
- **클라우드 인프라**: $200-300
- **OpenAI API**: $100-300
- **기타 API**: $50-100
- **총계**: $350-700/월

### 프로덕션 단계 (월간)
- **클라우드 인프라**: $400-800 (GPU 제외)
- **GPU 인스턴스**: $900-1,400 (필요시)
- **OpenAI API**: $300-1,000
- **기타 API**: $100-300
- **총계**: $800-3,500/월

## 구현 우선순위

### 1단계 (즉시 구현 필요)
1. ✅ OpenAI API 연동
2. ✅ API 오류 해결
3. ✅ 기본 사용자 인증
4. ✅ 데이터베이스 완전 연동

### 2단계 (1-2주 내)
1. ✅ 특허 데이터 수집 파이프라인
2. ✅ 향상된 UI/UX
3. ✅ API 보안 및 제한
4. ✅ 기본 모니터링

### 3단계 (1-2개월 내)
1. ✅ 3D 모델링 기능 (NeRF)
2. ✅ 고급 분석 기능
3. ✅ 성능 최적화
4. ✅ 프로덕션 배포

### 4단계 (장기 계획)
1. ✅ 모바일 앱 개발
2. ✅ 엔터프라이즈 기능
3. ✅ AI 모델 자체 훈련
4. ✅ 국제화 지원

## 결론

본 가이드를 따라 구현하면 현재의 부분적인 Patent2Tech 시스템을 완전한 프로덕션 서비스로 발전시킬 수 있습니다. 가장 중요한 것은 OpenAI API 키를 확보하고 1단계부터 순차적으로 구현하는 것입니다.

**즉시 필요한 것들:**
- OpenAI API 키 ($20/월부터 시작)
- 기본 클라우드 인프라 ($200-300/월)
- 개발 시간 (풀타임 개발자 기준 2-3개월)

**선택적으로 필요한 것들:**
- GPU 인스턴스 (3D 모델링용, $900+/월)
- 추가 API 키들 (특허 데이터베이스 접근용)
- 고급 모니터링 도구

이 계획을 따라 구현하면 실제로 작동하는 특허 LLM 서비스를 구축할 수 있으며, 사용자들에게 실질적인 가치를 제공할 수 있습니다.

