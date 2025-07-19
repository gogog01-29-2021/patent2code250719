# Patent2Tech LLM 시스템 완전 구현 계획

## 1. 필요한 API 키 및 리소스

### 1.1 필수 API 키
- **OpenAI API Key**: GPT-4 또는 GPT-3.5-turbo 모델 사용을 위해 필요
  - 용도: 특허 문서 분석, 기술 개념 추출, 코드 생성
  - 예상 비용: 월 $100-500 (사용량에 따라)
  
- **Hugging Face API Token**: PatentGPT 및 특허 특화 모델 사용
  - 용도: 특허 도메인 특화 LLM 모델 접근
  - 비용: 무료 (일부 모델은 유료)

### 1.2 특허 데이터 접근
- **USPTO API**: 미국 특허 데이터 접근 (무료)
- **KIPO API**: 한국 특허 데이터 접근 (무료)
- **EPO API**: 유럽 특허 데이터 접근 (무료)
- **Google Patents API**: 통합 특허 검색 (무료, 제한 있음)

### 1.3 3D 모델링 및 NeRF 관련
- **NVIDIA GPU 클라우드 크레딧**: NeRF 모델 훈련 및 추론용
- **3D 모델링 라이브러리**: Open3D, PyTorch3D (오픈소스)

## 2. 실제 LLM 연동 구현 방안

### 2.1 OpenAI API 통합
```python
# patent_llm_architecture.py 개선
import openai
from openai import OpenAI

class Patent2TechSystem:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def analyze_patent(self, patent_text):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a patent analysis expert..."},
                {"role": "user", "content": f"Analyze this patent: {patent_text}"}
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content
        
    def generate_implementation(self, tech_concept, impl_type):
        prompt = f"Generate {impl_type} implementation for: {tech_concept}"
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a technical implementation expert..."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000
        )
        return response.choices[0].message.content
```

### 2.2 PatentGPT 모델 통합
```python
# Hugging Face Transformers 사용
from transformers import AutoTokenizer, AutoModelForCausalLM

class PatentGPTIntegration:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("patent-gpt-model")
        self.model = AutoModelForCausalLM.from_pretrained("patent-gpt-model")
        
    def analyze_patent_claims(self, claims_text):
        inputs = self.tokenizer(claims_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=1000)
        return self.tokenizer.decode(outputs[0])
```

## 3. 특허 데이터 수집 파이프라인 구현

### 3.1 USPTO 데이터 수집
```python
import requests
import json

class USPTODataCollector:
    def __init__(self):
        self.base_url = "https://developer.uspto.gov/api/v1"
        
    def search_patents(self, query, limit=100):
        url = f"{self.base_url}/patent/search"
        params = {
            "query": query,
            "limit": limit,
            "format": "json"
        }
        response = requests.get(url, params=params)
        return response.json()
        
    def get_patent_details(self, patent_number):
        url = f"{self.base_url}/patent/{patent_number}"
        response = requests.get(url)
        return response.json()
```

### 3.2 데이터베이스 스키마 개선
```sql
-- 특허 데이터 테이블
CREATE TABLE patents (
    id INTEGER PRIMARY KEY,
    patent_number VARCHAR(50) UNIQUE,
    title TEXT,
    abstract TEXT,
    claims TEXT,
    description TEXT,
    inventors TEXT,
    assignee TEXT,
    filing_date DATE,
    publication_date DATE,
    classification_codes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 분석 결과 테이블
CREATE TABLE patent_analysis (
    id INTEGER PRIMARY KEY,
    patent_id INTEGER,
    tech_concepts TEXT,
    implementation_suggestions TEXT,
    code_examples TEXT,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patent_id) REFERENCES patents(id)
);

-- 3D 모델 데이터 테이블
CREATE TABLE patent_3d_models (
    id INTEGER PRIMARY KEY,
    patent_id INTEGER,
    model_file_path TEXT,
    model_type VARCHAR(50),
    generation_method VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patent_id) REFERENCES patents(id)
);
```

## 4. 3D 이미지 처리 및 NeRF 구현

### 4.1 NeRF 기반 3D 모델 생성
```python
import torch
import numpy as np
from nerf_pytorch import NeRF

class PatentDrawingTo3D:
    def __init__(self):
        self.nerf_model = NeRF()
        
    def process_patent_drawings(self, drawing_images):
        """특허 도면에서 3D 모델 생성"""
        # 이미지 전처리
        processed_images = self.preprocess_drawings(drawing_images)
        
        # NeRF 모델로 3D 재구성
        nerf_output = self.nerf_model.train(processed_images)
        
        # 3D 메시 생성
        mesh = self.generate_mesh(nerf_output)
        
        return mesh
        
    def preprocess_drawings(self, images):
        """특허 도면 전처리"""
        # 선 추출, 노이즈 제거, 정규화
        return processed_images
        
    def generate_mesh(self, nerf_output):
        """NeRF 출력에서 3D 메시 생성"""
        # Marching cubes 알고리즘 사용
        return mesh
```

### 4.2 CAD 파일 생성
```python
import cadquery as cq
from OCC.Core import TopoDS_Shape

class CADGenerator:
    def __init__(self):
        pass
        
    def mesh_to_cad(self, mesh):
        """3D 메시를 CAD 파일로 변환"""
        # 메시를 CAD 형식으로 변환
        cad_object = cq.Workplane().add(mesh)
        return cad_object
        
    def export_formats(self, cad_object, base_filename):
        """다양한 CAD 형식으로 내보내기"""
        formats = {
            'step': lambda: cad_object.val().exportStep(f"{base_filename}.step"),
            'stl': lambda: cad_object.val().exportStl(f"{base_filename}.stl"),
            'obj': lambda: cad_object.val().exportObj(f"{base_filename}.obj")
        }
        
        for format_name, export_func in formats.items():
            try:
                export_func()
            except Exception as e:
                print(f"Failed to export {format_name}: {e}")
```

## 5. 모델 훈련 및 파인튜닝 시스템

### 5.1 LoRA 파인튜닝 구현
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

class PatentLLMTrainer:
    def __init__(self, base_model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
    def setup_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.base_model, lora_config)
        
    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir="./patent_llm_lora",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
```

### 5.2 특허 데이터셋 준비
```python
class PatentDatasetBuilder:
    def __init__(self):
        self.patent_collector = USPTODataCollector()
        
    def build_training_dataset(self, domains, size_per_domain=1000):
        """특허 도메인별 훈련 데이터셋 구축"""
        dataset = []
        
        for domain in domains:
            patents = self.patent_collector.search_patents(
                query=f"classification:{domain}", 
                limit=size_per_domain
            )
            
            for patent in patents:
                # 특허 텍스트를 훈련 형식으로 변환
                training_example = self.format_for_training(patent)
                dataset.append(training_example)
                
        return dataset
        
    def format_for_training(self, patent):
        """특허 데이터를 훈련 형식으로 변환"""
        prompt = f"Patent Title: {patent['title']}\nAbstract: {patent['abstract']}\nClaims: {patent['claims']}"
        response = f"Technical Analysis: {patent['analysis']}\nImplementation: {patent['implementation']}"
        
        return {
            "input": prompt,
            "output": response
        }
```

## 6. 보안 및 인증 시스템

### 6.1 사용자 인증
```python
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.security import generate_password_hash, check_password_hash

class AuthSystem:
    def __init__(self, app):
        self.jwt = JWTManager(app)
        
    def register_user(self, username, password, email):
        hashed_password = generate_password_hash(password)
        # 데이터베이스에 사용자 저장
        
    def login_user(self, username, password):
        user = self.get_user(username)
        if user and check_password_hash(user.password, password):
            access_token = create_access_token(identity=username)
            return access_token
        return None
        
    @jwt_required()
    def protected_route(self):
        # 보호된 라우트 구현
        pass
```

### 6.2 API 키 관리
```python
import os
from cryptography.fernet import Fernet

class APIKeyManager:
    def __init__(self):
        self.cipher_suite = Fernet(os.environ.get('ENCRYPTION_KEY'))
        
    def encrypt_api_key(self, api_key):
        return self.cipher_suite.encrypt(api_key.encode())
        
    def decrypt_api_key(self, encrypted_key):
        return self.cipher_suite.decrypt(encrypted_key).decode()
```

## 7. 테스트 코드 구현

### 7.1 단위 테스트
```python
import unittest
from unittest.mock import patch, MagicMock

class TestPatent2TechSystem(unittest.TestCase):
    def setUp(self):
        self.system = Patent2TechSystem(api_key="test_key")
        
    @patch('openai.ChatCompletion.create')
    def test_analyze_patent(self, mock_openai):
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.choices[0].message.content = "Test analysis"
        
        result = self.system.analyze_patent("Test patent text")
        self.assertEqual(result, "Test analysis")
        
    def test_data_collection(self):
        collector = USPTODataCollector()
        # 실제 API 호출 없이 테스트
        pass
```

### 7.2 통합 테스트
```python
class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        # 특허 검색 -> 분석 -> 구현 생성 -> 3D 모델링 전체 파이프라인 테스트
        pass
```

## 8. 배포 및 서비스화

### 8.1 Docker 컨테이너화
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "src/main.py"]
```

### 8.2 환경 변수 설정
```bash
# .env 파일
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_hf_token
DATABASE_URL=postgresql://user:pass@localhost/patent_db
JWT_SECRET_KEY=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
```

### 8.3 클라우드 배포 (AWS/GCP)
```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=patent_db
      - POSTGRES_USER=patent_user
      - POSTGRES_PASSWORD=patent_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 9. 예상 비용 및 리소스

### 9.1 API 비용 (월간)
- OpenAI API: $200-800 (사용량에 따라)
- 클라우드 호스팅: $50-200
- 데이터베이스: $30-100
- 총 예상 비용: $280-1100/월

### 9.2 개발 시간
- LLM 연동: 2-3주
- 데이터 파이프라인: 3-4주
- 3D 모델링 (NeRF): 4-6주
- 보안/인증: 1-2주
- 테스트 및 배포: 2-3주
- 총 개발 기간: 12-18주

## 10. 즉시 시작 가능한 단계

1. **OpenAI API 키 발급** - 즉시 가능
2. **기존 Flask API 오류 수정** - 1-2일
3. **기본 LLM 연동 구현** - 1주
4. **특허 데이터 수집 파이프라인 구축** - 2주
5. **3D 모델링 연구 및 프로토타입** - 3-4주

이 계획을 통해 현재의 프로토타입을 완전한 서비스로 발전시킬 수 있습니다.



## 11. 특허 데이터 수집 파이프라인 구체화

### 11.1 USPTO API 통합
```python
import requests
import json
import time
from datetime import datetime, timedelta

class USPTODataCollector:
    def __init__(self):
        self.base_url = "https://developer.uspto.gov/ds-api"
        self.search_url = "https://data.uspto.gov/api/search"
        
    def search_patents_by_classification(self, classification_code, limit=100):
        """CPC 분류 코드로 특허 검색"""
        params = {
            "searchText": f"cpc_class_title:{classification_code}",
            "start": 0,
            "rows": limit,
            "format": "json"
        }
        
        response = requests.get(self.search_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"USPTO API Error: {response.status_code}")
    
    def get_patent_full_text(self, patent_number):
        """특허 전문 텍스트 가져오기"""
        url = f"{self.base_url}/patent/{patent_number}/fulltext"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    
    def get_patent_images(self, patent_number):
        """특허 도면 이미지 가져오기"""
        url = f"{self.base_url}/patent/{patent_number}/images"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    
    def bulk_download_patents(self, classification_codes, patents_per_class=1000):
        """대량 특허 데이터 다운로드"""
        all_patents = []
        
        for code in classification_codes:
            print(f"Downloading patents for classification: {code}")
            patents = self.search_patents_by_classification(code, patents_per_class)
            
            for patent in patents.get('results', []):
                # 상세 정보 가져오기
                full_text = self.get_patent_full_text(patent['patent_number'])
                images = self.get_patent_images(patent['patent_number'])
                
                patent_data = {
                    'basic_info': patent,
                    'full_text': full_text,
                    'images': images,
                    'classification': code,
                    'download_date': datetime.now().isoformat()
                }
                
                all_patents.append(patent_data)
                
                # API 제한 준수를 위한 지연
                time.sleep(0.1)
        
        return all_patents
```

### 11.2 EPO Open Patent Services (OPS) 통합
```python
import requests
import base64
from requests_oauthlib import OAuth2Session

class EPODataCollector:
    def __init__(self, consumer_key, consumer_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.base_url = "https://ops.epo.org/3.2/rest-services"
        self.token = None
        
    def authenticate(self):
        """EPO OPS OAuth 인증"""
        auth_url = "https://ops.epo.org/3.2/auth/accesstoken"
        
        # Base64 인코딩된 인증 정보
        credentials = base64.b64encode(
            f"{self.consumer_key}:{self.consumer_secret}".encode()
        ).decode()
        
        headers = {
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {'grant_type': 'client_credentials'}
        
        response = requests.post(auth_url, headers=headers, data=data)
        
        if response.status_code == 200:
            self.token = response.json()['access_token']
            return True
        else:
            raise Exception(f"EPO Authentication failed: {response.status_code}")
    
    def search_patents(self, query, range_begin=1, range_end=100):
        """EPO 특허 검색"""
        if not self.token:
            self.authenticate()
            
        url = f"{self.base_url}/published-data/search"
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/json'
        }
        
        params = {
            'q': query,
            'Range': f'{range_begin}-{range_end}'
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"EPO Search failed: {response.status_code}")
    
    def get_patent_biblio(self, patent_number):
        """특허 서지 정보 가져오기"""
        if not self.token:
            self.authenticate()
            
        url = f"{self.base_url}/published-data/publication/epodoc/{patent_number}/biblio"
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    
    def get_patent_fulltext(self, patent_number):
        """특허 전문 가져오기"""
        if not self.token:
            self.authenticate()
            
        url = f"{self.base_url}/published-data/publication/epodoc/{patent_number}/fulltext"
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
```

### 11.3 KIPO 데이터 수집 (KIPRIS Plus API)
```python
import requests
import xml.etree.ElementTree as ET

class KIPODataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://plus.kipris.or.kr/openapi/rest"
        
    def search_patents(self, query, start=1, display=100):
        """KIPO 특허 검색"""
        url = f"{self.base_url}/patUtiModInfoSearchSevice/patentUtilityModelSearch"
        
        params = {
            'accessKey': self.api_key,
            'query': query,
            'start': start,
            'display': display,
            'format': 'json'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"KIPO API Error: {response.status_code}")
    
    def get_patent_detail(self, application_number):
        """특허 상세 정보 가져오기"""
        url = f"{self.base_url}/patUtiModInfoSearchSevice/patentUtilityModelInfo"
        
        params = {
            'accessKey': self.api_key,
            'applicationNumber': application_number,
            'format': 'json'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
```

### 11.4 통합 데이터 수집 파이프라인
```python
import sqlite3
import json
from datetime import datetime
import threading
from queue import Queue

class IntegratedPatentCollector:
    def __init__(self, uspto_config=None, epo_config=None, kipo_config=None):
        self.uspto = USPTODataCollector() if uspto_config else None
        self.epo = EPODataCollector(**epo_config) if epo_config else None
        self.kipo = KIPODataCollector(kipo_config['api_key']) if kipo_config else None
        
        self.db_path = "patent_database.db"
        self.init_database()
        
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 특허 기본 정보 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patent_number VARCHAR(50) UNIQUE,
                title TEXT,
                abstract TEXT,
                claims TEXT,
                description TEXT,
                inventors TEXT,
                assignee TEXT,
                filing_date DATE,
                publication_date DATE,
                classification_codes TEXT,
                source VARCHAR(20),
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 특허 이미지 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patent_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patent_id INTEGER,
                image_url TEXT,
                image_type VARCHAR(50),
                local_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patent_id) REFERENCES patents(id)
            )
        ''')
        
        # 수집 작업 로그 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source VARCHAR(20),
                query TEXT,
                total_collected INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status VARCHAR(20),
                error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_by_technology_domains(self, domains, patents_per_domain=1000):
        """기술 도메인별 특허 수집"""
        collection_results = {}
        
        for domain in domains:
            print(f"Collecting patents for domain: {domain}")
            
            domain_results = {
                'uspto': [],
                'epo': [],
                'kipo': []
            }
            
            # USPTO 수집
            if self.uspto:
                try:
                    uspto_patents = self.uspto.search_patents_by_classification(
                        domain, patents_per_domain
                    )
                    domain_results['uspto'] = uspto_patents
                    self.save_patents_to_db(uspto_patents, 'USPTO')
                except Exception as e:
                    print(f"USPTO collection error for {domain}: {e}")
            
            # EPO 수집
            if self.epo:
                try:
                    epo_patents = self.epo.search_patents(
                        f"classification:{domain}", 1, patents_per_domain
                    )
                    domain_results['epo'] = epo_patents
                    self.save_patents_to_db(epo_patents, 'EPO')
                except Exception as e:
                    print(f"EPO collection error for {domain}: {e}")
            
            # KIPO 수집
            if self.kipo:
                try:
                    kipo_patents = self.kipo.search_patents(
                        domain, 1, patents_per_domain
                    )
                    domain_results['kipo'] = kipo_patents
                    self.save_patents_to_db(kipo_patents, 'KIPO')
                except Exception as e:
                    print(f"KIPO collection error for {domain}: {e}")
            
            collection_results[domain] = domain_results
        
        return collection_results
    
    def save_patents_to_db(self, patents_data, source):
        """특허 데이터를 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for patent in patents_data:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO patents 
                    (patent_number, title, abstract, claims, description, 
                     inventors, assignee, filing_date, publication_date, 
                     classification_codes, source, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patent.get('patent_number'),
                    patent.get('title'),
                    patent.get('abstract'),
                    patent.get('claims'),
                    patent.get('description'),
                    json.dumps(patent.get('inventors', [])),
                    patent.get('assignee'),
                    patent.get('filing_date'),
                    patent.get('publication_date'),
                    json.dumps(patent.get('classification_codes', [])),
                    source,
                    json.dumps(patent)
                ))
            except Exception as e:
                print(f"Error saving patent {patent.get('patent_number')}: {e}")
        
        conn.commit()
        conn.close()
    
    def get_collection_statistics(self):
        """수집 통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT source, COUNT(*) as count 
            FROM patents 
            GROUP BY source
        ''')
        
        stats = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM patents')
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_patents': total,
            'by_source': stats
        }
```

### 11.5 데이터 품질 관리 및 중복 제거
```python
import hashlib
from difflib import SequenceMatcher

class PatentDataQualityManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def detect_duplicates(self, similarity_threshold=0.9):
        """중복 특허 탐지"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, patent_number, title, abstract FROM patents')
        patents = cursor.fetchall()
        
        duplicates = []
        
        for i, patent1 in enumerate(patents):
            for j, patent2 in enumerate(patents[i+1:], i+1):
                # 제목과 초록 유사도 계산
                title_similarity = SequenceMatcher(
                    None, patent1[2] or '', patent2[2] or ''
                ).ratio()
                
                abstract_similarity = SequenceMatcher(
                    None, patent1[3] or '', patent2[3] or ''
                ).ratio()
                
                avg_similarity = (title_similarity + abstract_similarity) / 2
                
                if avg_similarity > similarity_threshold:
                    duplicates.append({
                        'patent1': {'id': patent1[0], 'number': patent1[1]},
                        'patent2': {'id': patent2[0], 'number': patent2[1]},
                        'similarity': avg_similarity
                    })
        
        conn.close()
        return duplicates
    
    def clean_patent_text(self, text):
        """특허 텍스트 정제"""
        if not text:
            return ""
        
        # HTML 태그 제거
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정규화
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def validate_patent_data(self, patent_data):
        """특허 데이터 유효성 검증"""
        required_fields = ['patent_number', 'title']
        
        for field in required_fields:
            if not patent_data.get(field):
                return False, f"Missing required field: {field}"
        
        # 특허 번호 형식 검증
        patent_number = patent_data['patent_number']
        if not re.match(r'^[A-Z]{2}\d+[A-Z]?\d*$', patent_number):
            return False, f"Invalid patent number format: {patent_number}"
        
        return True, "Valid"
```

### 11.6 실시간 데이터 수집 스케줄러
```python
import schedule
import time
from datetime import datetime, timedelta

class PatentDataScheduler:
    def __init__(self, collector):
        self.collector = collector
        
    def daily_collection_job(self):
        """일일 특허 데이터 수집 작업"""
        print(f"Starting daily collection job at {datetime.now()}")
        
        # 최근 일주일간 공개된 특허 수집
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        try:
            # 각 소스별로 최신 특허 수집
            results = self.collector.collect_recent_patents(start_date, end_date)
            
            print(f"Daily collection completed. Collected {len(results)} patents")
            
        except Exception as e:
            print(f"Daily collection failed: {e}")
    
    def weekly_cleanup_job(self):
        """주간 데이터 정리 작업"""
        print(f"Starting weekly cleanup job at {datetime.now()}")
        
        quality_manager = PatentDataQualityManager(self.collector.db_path)
        
        # 중복 제거
        duplicates = quality_manager.detect_duplicates()
        print(f"Found {len(duplicates)} potential duplicates")
        
        # 데이터 품질 검증
        # ... 추가 정리 작업
    
    def start_scheduler(self):
        """스케줄러 시작"""
        # 매일 오전 2시에 데이터 수집
        schedule.every().day.at("02:00").do(self.daily_collection_job)
        
        # 매주 일요일 오전 3시에 정리 작업
        schedule.every().sunday.at("03:00").do(self.weekly_cleanup_job)
        
        print("Patent data scheduler started")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크
```

이러한 구체적인 구현을 통해 USPTO, EPO, KIPO 등 주요 특허청의 데이터를 체계적으로 수집하고 관리할 수 있습니다.


## 12. 3D 이미지 처리 모듈 (NeRF 포함) 구현 계획

### 12.1 NeRF 기반 특허 도면 3D 변환 시스템
```python
import torch
import torch.nn as nn
import numpy as np
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.viewer.server.viewer_state import ViewerState
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class PatentDrawingNeRFProcessor:
    """특허 도면을 3D 모델로 변환하는 NeRF 기반 프로세서"""
    
    def __init__(self, config_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.datamanager = None
        
    def preprocess_patent_drawing(self, image_path):
        """특허 도면 전처리"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 5)
        
        # 엣지 검출
        edges = cv2.Canny(denoised, 50, 150)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 주요 객체 분리
        main_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 최소 면적 필터
                main_objects.append(contour)
        
        return {
            'original': image,
            'preprocessed': denoised,
            'edges': edges,
            'contours': main_objects
        }
    
    def generate_multi_view_images(self, processed_data, num_views=8):
        """단일 특허 도면에서 다중 뷰 이미지 생성"""
        original = processed_data['original']
        contours = processed_data['contours']
        
        # 깊이 정보 추정
        depth_map = self.estimate_depth_from_drawing(original, contours)
        
        # 다중 뷰 생성
        views = []
        angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
        
        for i, angle in enumerate(angles):
            # 회전 변환 행렬
            rotation_matrix = cv2.getRotationMatrix2D(
                (original.shape[1]//2, original.shape[0]//2), 
                np.degrees(angle), 
                1.0
            )
            
            # 이미지 회전
            rotated = cv2.warpAffine(original, rotation_matrix, 
                                   (original.shape[1], original.shape[0]))
            
            # 깊이 기반 변형 적용
            transformed = self.apply_depth_transformation(rotated, depth_map, angle)
            
            views.append({
                'image': transformed,
                'angle': angle,
                'camera_pose': self.calculate_camera_pose(angle)
            })
        
        return views
    
    def estimate_depth_from_drawing(self, image, contours):
        """특허 도면에서 깊이 정보 추정"""
        height, width = image.shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # 윤곽선 기반 깊이 추정
        for i, contour in enumerate(contours):
            # 윤곽선 마스크 생성
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # 면적에 따른 깊이 할당 (큰 객체일수록 앞쪽)
            area = cv2.contourArea(contour)
            depth_value = 1.0 - (area / (width * height))  # 정규화된 깊이
            
            depth_map[mask > 0] = depth_value
        
        # 가우시안 블러로 부드럽게
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        
        return depth_map
    
    def apply_depth_transformation(self, image, depth_map, angle):
        """깊이 정보를 이용한 이미지 변형"""
        height, width = image.shape[:2]
        
        # 원근 변환 매트릭스 계산
        perspective_strength = 0.3 * np.sin(angle)
        
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        dst_points = np.float32([
            [perspective_strength * width * 0.1, 0],
            [width - perspective_strength * width * 0.1, 0],
            [width, height],
            [0, height]
        ])
        
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(image, perspective_matrix, (width, height))
        
        return transformed
    
    def calculate_camera_pose(self, angle):
        """카메라 포즈 계산"""
        radius = 2.0
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0
        
        # 카메라 위치
        camera_position = np.array([x, y, z])
        
        # 타겟 (원점)
        target = np.array([0, 0, 0])
        
        # 업 벡터
        up = np.array([0, 0, 1])
        
        # 뷰 매트릭스 계산
        forward = target - camera_position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # 4x4 변환 매트릭스
        pose_matrix = np.eye(4)
        pose_matrix[:3, 0] = right
        pose_matrix[:3, 1] = up
        pose_matrix[:3, 2] = -forward
        pose_matrix[:3, 3] = camera_position
        
        return pose_matrix
    
    def train_nerf_model(self, views, num_iterations=5000):
        """NeRF 모델 훈련"""
        from nerfstudio.configs.method_configs import method_configs
        from nerfstudio.engine.trainer import Trainer
        
        # 데이터 준비
        train_data = self.prepare_training_data(views)
        
        # 설정
        config = method_configs["nerfacto"]
        config.machine.num_gpus = 1 if torch.cuda.is_available() else 0
        config.trainer.max_num_iterations = num_iterations
        
        # 트레이너 초기화
        trainer = Trainer(config, local_rank=0, world_size=1)
        
        # 훈련 실행
        trainer.setup()
        trainer.train()
        
        return trainer.pipeline.model
    
    def prepare_training_data(self, views):
        """NeRF 훈련용 데이터 준비"""
        images = []
        poses = []
        
        for view in views:
            # 이미지 정규화
            img = view['image'].astype(np.float32) / 255.0
            images.append(img)
            
            # 포즈 매트릭스
            poses.append(view['camera_pose'])
        
        return {
            'images': np.array(images),
            'poses': np.array(poses),
            'intrinsics': self.get_default_intrinsics()
        }
    
    def get_default_intrinsics(self):
        """기본 카메라 내부 파라미터"""
        return np.array([
            [800, 0, 400],
            [0, 800, 300],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def render_3d_model(self, model, output_path, num_frames=36):
        """3D 모델 렌더링"""
        frames = []
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            camera_pose = self.calculate_camera_pose(angle)
            
            # 렌더링
            with torch.no_grad():
                rendered = model.get_outputs_for_camera_ray_bundle(
                    self.create_camera_ray_bundle(camera_pose)
                )
            
            frame = rendered['rgb'].cpu().numpy()
            frames.append(frame)
        
        # 비디오 저장
        self.save_video(frames, output_path)
        
        return frames
    
    def create_camera_ray_bundle(self, camera_pose):
        """카메라 레이 번들 생성"""
        # 구현 필요: nerfstudio의 카메라 레이 번들 생성
        pass
    
    def save_video(self, frames, output_path):
        """프레임들을 비디오로 저장"""
        import imageio
        
        # 프레임을 0-255 범위로 변환
        frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]
        
        # 비디오 저장
        imageio.mimsave(output_path, frames_uint8, fps=10)
        
        print(f"3D model video saved to {output_path}")
```

### 12.2 Instant-NGP 기반 고속 3D 재구성
```python
import torch
import torch.nn as nn
import tinycudann as tcnn
from typing import Dict, List, Optional

class InstantNGPPatentProcessor:
    """Instant-NGP를 사용한 고속 특허 도면 3D 재구성"""
    
    def __init__(self, 
                 hash_encoding_config: Optional[Dict] = None,
                 mlp_config: Optional[Dict] = None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 기본 해시 인코딩 설정
        self.hash_encoding_config = hash_encoding_config or {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5
        }
        
        # 기본 MLP 설정
        self.mlp_config = mlp_config or {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        }
        
        self.setup_networks()
    
    def setup_networks(self):
        """네트워크 초기화"""
        # 위치 인코딩
        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=self.hash_encoding_config
        )
        
        # 방향 인코딩
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4
            }
        )
        
        # 밀도 네트워크
        self.density_network = tcnn.Network(
            n_input_dims=self.position_encoding.n_output_dims,
            n_output_dims=16,
            network_config=self.mlp_config
        )
        
        # 색상 네트워크
        color_network_config = self.mlp_config.copy()
        color_network_config["n_output_dims"] = 3
        
        self.color_network = tcnn.Network(
            n_input_dims=16 + self.direction_encoding.n_output_dims,
            n_output_dims=3,
            network_config=color_network_config
        )
    
    def process_patent_drawing_fast(self, image_path: str, 
                                  training_steps: int = 1000) -> Dict:
        """고속 특허 도면 처리"""
        
        # 1. 이미지 전처리
        processed_data = self.preprocess_patent_image(image_path)
        
        # 2. 다중 뷰 생성 (간소화된 버전)
        views = self.generate_synthetic_views(processed_data, num_views=12)
        
        # 3. 고속 훈련
        model = self.fast_train(views, training_steps)
        
        # 4. 3D 메시 추출
        mesh = self.extract_mesh(model)
        
        return {
            'model': model,
            'mesh': mesh,
            'views': views,
            'processed_data': processed_data
        }
    
    def preprocess_patent_image(self, image_path: str) -> Dict:
        """특허 이미지 전처리 (최적화된 버전)"""
        import cv2
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # 크기 정규화
        target_size = (512, 512)
        resized = cv2.resize(image, target_size)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 적응적 임계값
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return {
            'original': resized,
            'binary': cleaned,
            'gray': gray
        }
    
    def generate_synthetic_views(self, processed_data: Dict, 
                               num_views: int = 12) -> List[Dict]:
        """합성 뷰 생성 (고속 버전)"""
        views = []
        original = processed_data['original']
        
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views
            
            # 간단한 원근 변환
            view_image = self.apply_perspective_transform(original, angle)
            
            # 카메라 포즈
            pose = self.calculate_simple_pose(angle)
            
            views.append({
                'image': view_image,
                'pose': pose,
                'angle': angle
            })
        
        return views
    
    def apply_perspective_transform(self, image: np.ndarray, 
                                  angle: float) -> np.ndarray:
        """간단한 원근 변환"""
        height, width = image.shape[:2]
        
        # 변환 강도
        strength = 0.2 * np.sin(angle)
        
        src_pts = np.float32([
            [0, 0], [width, 0], [width, height], [0, height]
        ])
        
        dst_pts = np.float32([
            [strength * width * 0.1, 0],
            [width - strength * width * 0.1, 0],
            [width, height],
            [0, height]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        transformed = cv2.warpPerspective(image, matrix, (width, height))
        
        return transformed
    
    def calculate_simple_pose(self, angle: float) -> np.ndarray:
        """간단한 포즈 계산"""
        radius = 1.5
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.5
        
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        
        return pose
    
    def fast_train(self, views: List[Dict], steps: int) -> nn.Module:
        """고속 훈련"""
        optimizer = torch.optim.Adam([
            {'params': self.position_encoding.parameters()},
            {'params': self.density_network.parameters()},
            {'params': self.color_network.parameters()}
        ], lr=1e-2)
        
        for step in range(steps):
            # 랜덤 뷰 선택
            view = np.random.choice(views)
            
            # 레이 샘플링
            rays_o, rays_d = self.sample_rays(view)
            
            # 포워드 패스
            rgb, density = self.forward(rays_o, rays_d)
            
            # 손실 계산
            target_rgb = self.get_target_rgb(view, rays_o, rays_d)
            loss = nn.MSELoss()(rgb, target_rgb)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")
        
        return self
    
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """순전파"""
        # 레이 상의 점들 샘플링
        t_vals = torch.linspace(0.0, 1.0, 64, device=self.device)
        points = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]
        
        # 위치 인코딩
        points_flat = points.reshape(-1, 3)
        encoded_positions = self.position_encoding(points_flat)
        
        # 밀도 예측
        density_features = self.density_network(encoded_positions)
        density = density_features[..., 0]
        
        # 방향 인코딩
        dirs_flat = rays_d.unsqueeze(-2).expand_as(points).reshape(-1, 3)
        encoded_directions = self.direction_encoding(dirs_flat)
        
        # 색상 예측
        color_input = torch.cat([density_features[..., 1:], encoded_directions], dim=-1)
        rgb = self.color_network(color_input)
        
        # 볼륨 렌더링
        rgb = rgb.reshape(*points.shape[:-1], 3)
        density = density.reshape(*points.shape[:-1])
        
        # 알파 합성
        alpha = 1.0 - torch.exp(-density * 0.01)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha], dim=-1), 
            dim=-1
        )[..., :-1]
        
        rgb_final = torch.sum(weights[..., None] * rgb, dim=-2)
        
        return rgb_final, density
    
    def sample_rays(self, view: Dict) -> tuple:
        """레이 샘플링"""
        image = view['image']
        pose = view['pose']
        
        height, width = image.shape[:2]
        
        # 랜덤 픽셀 선택
        num_rays = 1024
        i = torch.randint(0, height, (num_rays,), device=self.device)
        j = torch.randint(0, width, (num_rays,), device=self.device)
        
        # 카메라 좌표계에서 레이 방향 계산
        focal = 400.0  # 가정된 초점 거리
        
        dirs = torch.stack([
            (j - width * 0.5) / focal,
            -(i - height * 0.5) / focal,
            -torch.ones_like(i, dtype=torch.float32)
        ], dim=-1)
        
        # 월드 좌표계로 변환
        pose_tensor = torch.from_numpy(pose).float().to(self.device)
        rays_d = torch.sum(dirs[..., None, :] * pose_tensor[:3, :3], dim=-1)
        rays_o = pose_tensor[:3, 3].expand(rays_d.shape)
        
        return rays_o, rays_d
    
    def get_target_rgb(self, view: Dict, rays_o: torch.Tensor, 
                      rays_d: torch.Tensor) -> torch.Tensor:
        """타겟 RGB 값 추출"""
        # 간단한 구현: 이미지에서 해당 픽셀 값 추출
        image = view['image']
        height, width = image.shape[:2]
        
        # 레이에 해당하는 픽셀 좌표 계산 (역변환)
        # 실제로는 더 정확한 계산이 필요
        num_rays = rays_o.shape[0]
        target_rgb = torch.rand(num_rays, 3, device=self.device)  # 임시
        
        return target_rgb
    
    def extract_mesh(self, model) -> Dict:
        """3D 메시 추출"""
        # 마칭 큐브 알고리즘을 사용한 메시 추출
        resolution = 128
        
        # 3D 그리드 생성
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        z = torch.linspace(-1, 1, resolution)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # 밀도 값 계산
        with torch.no_grad():
            encoded = self.position_encoding(points.to(self.device))
            density_features = self.density_network(encoded)
            density = density_features[..., 0]
        
        density_grid = density.reshape(resolution, resolution, resolution)
        
        # 마칭 큐브 (간단한 구현)
        vertices, faces = self.marching_cubes(density_grid.cpu().numpy())
        
        return {
            'vertices': vertices,
            'faces': faces,
            'density_grid': density_grid
        }
    
    def marching_cubes(self, density_grid: np.ndarray, threshold: float = 0.5):
        """마칭 큐브 알고리즘 (간단한 구현)"""
        try:
            from skimage import measure
            vertices, faces, _, _ = measure.marching_cubes(
                density_grid, level=threshold
            )
            return vertices, faces
        except ImportError:
            print("scikit-image not available, returning dummy mesh")
            return np.array([[0, 0, 0]]), np.array([[0, 0, 0]])
```

### 12.3 특허 도면 특화 3D 재구성 파이프라인
```python
class PatentSpecific3DPipeline:
    """특허 도면에 특화된 3D 재구성 파이프라인"""
    
    def __init__(self):
        self.nerf_processor = PatentDrawingNeRFProcessor()
        self.instant_ngp = InstantNGPPatentProcessor()
        
    def analyze_patent_drawing_type(self, image_path: str) -> str:
        """특허 도면 타입 분석"""
        import cv2
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 선의 밀도 분석
        edges = cv2.Canny(image, 50, 150)
        line_density = np.sum(edges > 0) / edges.size
        
        # 원형 객체 검출
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        has_circles = circles is not None and len(circles[0]) > 0
        
        # 직선 검출
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        
        has_lines = lines is not None and len(lines) > 10
        
        # 도면 타입 분류
        if has_circles and line_density > 0.1:
            return "mechanical"  # 기계 도면
        elif has_lines and line_density > 0.05:
            return "architectural"  # 건축 도면
        elif line_density < 0.03:
            return "schematic"  # 회로도/개념도
        else:
            return "general"  # 일반 도면
    
    def process_by_type(self, image_path: str, drawing_type: str = None) -> Dict:
        """도면 타입에 따른 처리"""
        if drawing_type is None:
            drawing_type = self.analyze_patent_drawing_type(image_path)
        
        print(f"Detected drawing type: {drawing_type}")
        
        if drawing_type == "mechanical":
            return self.process_mechanical_drawing(image_path)
        elif drawing_type == "architectural":
            return self.process_architectural_drawing(image_path)
        elif drawing_type == "schematic":
            return self.process_schematic_drawing(image_path)
        else:
            return self.process_general_drawing(image_path)
    
    def process_mechanical_drawing(self, image_path: str) -> Dict:
        """기계 도면 처리"""
        # 고정밀 NeRF 사용
        processed_data = self.nerf_processor.preprocess_patent_drawing(image_path)
        views = self.nerf_processor.generate_multi_view_images(processed_data, num_views=16)
        model = self.nerf_processor.train_nerf_model(views, num_iterations=8000)
        
        return {
            'type': 'mechanical',
            'model': model,
            'views': views,
            'method': 'high_precision_nerf'
        }
    
    def process_architectural_drawing(self, image_path: str) -> Dict:
        """건축 도면 처리"""
        # Instant-NGP 사용 (빠른 처리)
        result = self.instant_ngp.process_patent_drawing_fast(
            image_path, training_steps=2000
        )
        
        return {
            'type': 'architectural',
            'model': result['model'],
            'mesh': result['mesh'],
            'method': 'instant_ngp'
        }
    
    def process_schematic_drawing(self, image_path: str) -> Dict:
        """회로도/개념도 처리"""
        # 2.5D 표현 사용
        processed_data = self.create_2_5d_representation(image_path)
        
        return {
            'type': 'schematic',
            'representation': processed_data,
            'method': '2_5d_layered'
        }
    
    def process_general_drawing(self, image_path: str) -> Dict:
        """일반 도면 처리"""
        # 적응적 방법 선택
        result = self.instant_ngp.process_patent_drawing_fast(
            image_path, training_steps=1500
        )
        
        return {
            'type': 'general',
            'model': result['model'],
            'mesh': result['mesh'],
            'method': 'adaptive_ngp'
        }
    
    def create_2_5d_representation(self, image_path: str) -> Dict:
        """2.5D 레이어드 표현 생성"""
        import cv2
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 레이어 분리
        layers = []
        
        # 배경 레이어
        background = cv2.GaussianBlur(gray, (21, 21), 0)
        layers.append({
            'type': 'background',
            'image': background,
            'depth': 0.0
        })
        
        # 주요 객체 레이어
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 500:
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [contour], 255)
                
                object_layer = cv2.bitwise_and(gray, mask)
                
                layers.append({
                    'type': f'object_{i}',
                    'image': object_layer,
                    'depth': 0.1 * (i + 1),
                    'contour': contour
                })
        
        return {
            'layers': layers,
            'original': image
        }
    
    def export_3d_model(self, result: Dict, output_path: str, format: str = 'obj'):
        """3D 모델 내보내기"""
        if 'mesh' in result:
            vertices = result['mesh']['vertices']
            faces = result['mesh']['faces']
            
            if format.lower() == 'obj':
                self.save_obj(vertices, faces, output_path)
            elif format.lower() == 'ply':
                self.save_ply(vertices, faces, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        elif 'representation' in result:
            # 2.5D 표현을 3D 모델로 변환
            self.convert_2_5d_to_3d(result['representation'], output_path)
        
        else:
            print("No exportable 3D data found")
    
    def save_obj(self, vertices: np.ndarray, faces: np.ndarray, filepath: str):
        """OBJ 파일로 저장"""
        with open(filepath, 'w') as f:
            # 버텍스 쓰기
            for vertex in vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # 면 쓰기
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"3D model saved as OBJ: {filepath}")
    
    def save_ply(self, vertices: np.ndarray, faces: np.ndarray, filepath: str):
        """PLY 파일로 저장"""
        with open(filepath, 'w') as f:
            # 헤더
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # 버텍스 데이터
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # 면 데이터
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"3D model saved as PLY: {filepath}")
    
    def convert_2_5d_to_3d(self, representation: Dict, output_path: str):
        """2.5D 표현을 3D 모델로 변환"""
        layers = representation['layers']
        
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for layer in layers:
            if 'contour' in layer:
                # 윤곽선을 3D 메시로 변환
                contour = layer['contour']
                depth = layer['depth']
                
                # 상단 면 버텍스
                top_vertices = []
                for point in contour.reshape(-1, 2):
                    top_vertices.append([point[0], point[1], depth])
                
                # 하단 면 버텍스
                bottom_vertices = []
                for point in contour.reshape(-1, 2):
                    bottom_vertices.append([point[0], point[1], 0])
                
                vertices = np.array(top_vertices + bottom_vertices)
                
                # 면 생성
                n_points = len(top_vertices)
                faces = []
                
                # 측면 면들
                for i in range(n_points):
                    next_i = (i + 1) % n_points
                    
                    # 삼각형 1
                    faces.append([
                        vertex_offset + i,
                        vertex_offset + next_i,
                        vertex_offset + n_points + i
                    ])
                    
                    # 삼각형 2
                    faces.append([
                        vertex_offset + next_i,
                        vertex_offset + n_points + next_i,
                        vertex_offset + n_points + i
                    ])
                
                all_vertices.extend(vertices)
                all_faces.extend(faces)
                vertex_offset += len(vertices)
        
        # 저장
        if all_vertices:
            vertices_array = np.array(all_vertices)
            faces_array = np.array(all_faces)
            self.save_obj(vertices_array, faces_array, output_path)
        else:
            print("No 3D geometry generated from 2.5D representation")
```

### 12.4 통합 3D 처리 API
```python
class Patent3DProcessingAPI:
    """특허 3D 처리 통합 API"""
    
    def __init__(self):
        self.pipeline = PatentSpecific3DPipeline()
        
    def process_patent_to_3d(self, 
                           image_path: str,
                           output_dir: str,
                           options: Dict = None) -> Dict:
        """특허 도면을 3D 모델로 변환"""
        
        options = options or {}
        
        # 1. 도면 타입 분석
        drawing_type = self.pipeline.analyze_patent_drawing_type(image_path)
        
        # 2. 타입별 처리
        result = self.pipeline.process_by_type(image_path, drawing_type)
        
        # 3. 결과 저장
        output_files = {}
        
        # 3D 모델 내보내기
        if options.get('export_obj', True):
            obj_path = f"{output_dir}/model.obj"
            self.pipeline.export_3d_model(result, obj_path, 'obj')
            output_files['obj'] = obj_path
        
        if options.get('export_ply', False):
            ply_path = f"{output_dir}/model.ply"
            self.pipeline.export_3d_model(result, ply_path, 'ply')
            output_files['ply'] = ply_path
        
        # 렌더링 비디오 생성
        if options.get('create_video', True) and 'model' in result:
            video_path = f"{output_dir}/rotation.mp4"
            frames = self.pipeline.nerf_processor.render_3d_model(
                result['model'], video_path
            )
            output_files['video'] = video_path
        
        return {
            'drawing_type': drawing_type,
            'processing_method': result['method'],
            'output_files': output_files,
            'success': True
        }
    
    def batch_process(self, 
                     image_paths: List[str],
                     output_base_dir: str,
                     options: Dict = None) -> List[Dict]:
        """배치 처리"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            output_dir = f"{output_base_dir}/patent_{i:04d}"
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                result = self.process_patent_to_3d(image_path, output_dir, options)
                result['input_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results
```

이러한 구현을 통해 특허 도면을 다양한 방법으로 3D 모델로 변환할 수 있습니다. NeRF와 Instant-NGP를 활용하여 고품질의 3D 재구성이 가능하며, 특허 도면의 특성에 맞는 전처리와 후처리를 제공합니다.


## 13. 모델 훈련 및 파인튜닝 시스템 고도화 계획

### 13.1 특허 도메인 특화 훈련 전략

#### A. 다단계 훈련 파이프라인
```python
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging

class PatentLLMTrainingPipeline:
    """특허 LLM 훈련을 위한 다단계 파이프라인"""
    
    def __init__(self, 
                 base_model_name: str = "microsoft/DialoGPT-medium",
                 output_dir: str = "./patent_llm_outputs"):
        
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 모델 및 토크나이저 초기화
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # 훈련 단계별 설정
        self.training_stages = {
            'continual_pretraining': {
                'description': '특허 도메인 연속 사전훈련',
                'epochs': 3,
                'learning_rate': 5e-5,
                'batch_size': 8
            },
            'supervised_finetuning': {
                'description': '지도 학습 파인튜닝',
                'epochs': 5,
                'learning_rate': 2e-5,
                'batch_size': 4
            },
            'instruction_tuning': {
                'description': '인스트럭션 튜닝',
                'epochs': 3,
                'learning_rate': 1e-5,
                'batch_size': 2
            },
            'rlhf': {
                'description': '인간 피드백 강화학습',
                'epochs': 2,
                'learning_rate': 5e-6,
                'batch_size': 1
            }
        }
    
    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        self.logger.info(f"Loading base model: {self.base_model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 특허 도메인 특수 토큰 추가
        special_tokens = [
            "<patent_start>", "<patent_end>",
            "<claim_start>", "<claim_end>",
            "<abstract_start>", "<abstract_end>",
            "<description_start>", "<description_end>",
            "<figure_start>", "<figure_end>"
        ]
        
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.logger.info("Model and tokenizer setup completed")
    
    def setup_lora_config(self, stage: str) -> LoraConfig:
        """단계별 LoRA 설정"""
        stage_configs = {
            'continual_pretraining': {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ["q_proj", "v_proj"],
                'lora_dropout': 0.1
            },
            'supervised_finetuning': {
                'r': 32,
                'lora_alpha': 64,
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
                'lora_dropout': 0.05
            },
            'instruction_tuning': {
                'r': 64,
                'lora_alpha': 128,
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", 
                                 "gate_proj", "up_proj", "down_proj"],
                'lora_dropout': 0.05
            },
            'rlhf': {
                'r': 32,
                'lora_alpha': 64,
                'target_modules': ["q_proj", "v_proj"],
                'lora_dropout': 0.1
            }
        }
        
        config = stage_configs.get(stage, stage_configs['supervised_finetuning'])
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['r'],
            lora_alpha=config['lora_alpha'],
            target_modules=config['target_modules'],
            lora_dropout=config['lora_dropout'],
            bias="none"
        )
    
    def stage1_continual_pretraining(self, patent_corpus_path: str):
        """1단계: 특허 도메인 연속 사전훈련"""
        self.logger.info("Starting Stage 1: Continual Pre-training")
        
        # LoRA 설정
        lora_config = self.setup_lora_config('continual_pretraining')
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # 데이터 로드 및 전처리
        dataset = self.load_patent_corpus(patent_corpus_path)
        
        # 훈련 설정
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/stage1_continual_pretraining",
            num_train_epochs=self.training_stages['continual_pretraining']['epochs'],
            per_device_train_batch_size=self.training_stages['continual_pretraining']['batch_size'],
            learning_rate=self.training_stages['continual_pretraining']['learning_rate'],
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_drop_last=True,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="wandb" if wandb.api.api_key else None
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 트레이너 초기화
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 훈련 실행
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        self.tokenizer.save_pretrained(f"{self.output_dir}/stage1_continual_pretraining")
        
        self.logger.info("Stage 1 completed")
        return trainer
    
    def stage2_supervised_finetuning(self, patent_qa_dataset_path: str):
        """2단계: 지도 학습 파인튜닝"""
        self.logger.info("Starting Stage 2: Supervised Fine-tuning")
        
        # 이전 단계 모델 로드
        if self.peft_model is None:
            self.load_stage_model('stage1_continual_pretraining')
        
        # 새로운 LoRA 어댑터 추가
        lora_config = self.setup_lora_config('supervised_finetuning')
        self.peft_model.add_adapter("sft", lora_config)
        self.peft_model.set_adapter("sft")
        
        # 특허 Q&A 데이터셋 로드
        dataset = self.load_patent_qa_dataset(patent_qa_dataset_path)
        
        # 훈련 설정
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/stage2_supervised_finetuning",
            num_train_epochs=self.training_stages['supervised_finetuning']['epochs'],
            per_device_train_batch_size=self.training_stages['supervised_finetuning']['batch_size'],
            learning_rate=self.training_stages['supervised_finetuning']['learning_rate'],
            warmup_steps=200,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=250,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="wandb" if wandb.api.api_key else None
        )
        
        # 트레이너 초기화
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_qa_metrics
        )
        
        # 훈련 실행
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        
        self.logger.info("Stage 2 completed")
        return trainer
    
    def stage3_instruction_tuning(self, instruction_dataset_path: str):
        """3단계: 인스트럭션 튜닝"""
        self.logger.info("Starting Stage 3: Instruction Tuning")
        
        # 인스트럭션 데이터셋 로드
        dataset = self.load_instruction_dataset(instruction_dataset_path)
        
        # 새로운 LoRA 어댑터 추가
        lora_config = self.setup_lora_config('instruction_tuning')
        self.peft_model.add_adapter("instruction", lora_config)
        self.peft_model.set_adapter("instruction")
        
        # 훈련 설정
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/stage3_instruction_tuning",
            num_train_epochs=self.training_stages['instruction_tuning']['epochs'],
            per_device_train_batch_size=self.training_stages['instruction_tuning']['batch_size'],
            learning_rate=self.training_stages['instruction_tuning']['learning_rate'],
            warmup_steps=100,
            logging_steps=25,
            save_steps=250,
            evaluation_strategy="steps",
            eval_steps=125,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="wandb" if wandb.api.api_key else None
        )
        
        # 트레이너 초기화
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_instruction_metrics
        )
        
        # 훈련 실행
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        
        self.logger.info("Stage 3 completed")
        return trainer
    
    def stage4_rlhf_training(self, preference_dataset_path: str):
        """4단계: 인간 피드백 강화학습 (RLHF)"""
        self.logger.info("Starting Stage 4: RLHF Training")
        
        # 선호도 데이터셋 로드
        preference_dataset = self.load_preference_dataset(preference_dataset_path)
        
        # 보상 모델 훈련
        reward_model = self.train_reward_model(preference_dataset)
        
        # PPO 훈련
        ppo_trainer = self.setup_ppo_trainer(reward_model)
        ppo_trainer.train()
        
        self.logger.info("Stage 4 completed")
        return ppo_trainer
    
    def load_patent_corpus(self, corpus_path: str) -> Dict:
        """특허 코퍼스 로드 및 전처리"""
        # 특허 문서 로드
        with open(corpus_path, 'r', encoding='utf-8') as f:
            patent_texts = [line.strip() for line in f if line.strip()]
        
        # 토크나이징
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        # 데이터셋 생성
        dataset = Dataset.from_dict({'text': patent_texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 훈련/검증 분할
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        
        return {
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        }
    
    def load_patent_qa_dataset(self, dataset_path: str) -> Dict:
        """특허 Q&A 데이터셋 로드"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # 프롬프트 템플릿
        def format_qa_prompt(question: str, answer: str) -> str:
            return f"""<patent_start>
질문: {question}

답변: {answer}
<patent_end>"""
        
        # 데이터 포맷팅
        formatted_texts = []
        for item in qa_data:
            formatted_text = format_qa_prompt(item['question'], item['answer'])
            formatted_texts.append(formatted_text)
        
        # 토크나이징
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_dict({'text': formatted_texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 훈련/검증 분할
        split_dataset = tokenized_dataset.train_test_split(test_size=0.15)
        
        return {
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        }
    
    def load_instruction_dataset(self, dataset_path: str) -> Dict:
        """인스트럭션 데이터셋 로드"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            instruction_data = json.load(f)
        
        # 인스트럭션 프롬프트 템플릿
        def format_instruction_prompt(instruction: str, input_text: str, output: str) -> str:
            if input_text:
                return f"""다음은 특허 분석 작업을 설명하는 지시사항입니다. 요청을 적절히 완료하는 응답을 작성해주세요.

### 지시사항:
{instruction}

### 입력:
{input_text}

### 응답:
{output}"""
            else:
                return f"""다음은 특허 분석 작업을 설명하는 지시사항입니다. 요청을 적절히 완료하는 응답을 작성해주세요.

### 지시사항:
{instruction}

### 응답:
{output}"""
        
        # 데이터 포맷팅
        formatted_texts = []
        for item in instruction_data:
            formatted_text = format_instruction_prompt(
                item['instruction'], 
                item.get('input', ''), 
                item['output']
            )
            formatted_texts.append(formatted_text)
        
        # 토크나이징
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_dict({'text': formatted_texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 훈련/검증 분할
        split_dataset = tokenized_dataset.train_test_split(test_size=0.15)
        
        return {
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        }
    
    def load_preference_dataset(self, dataset_path: str) -> Dict:
        """선호도 데이터셋 로드 (RLHF용)"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            preference_data = json.load(f)
        
        # 선호도 데이터 포맷:
        # {
        #   "prompt": "특허 분석 요청",
        #   "chosen": "선호되는 응답",
        #   "rejected": "선호되지 않는 응답"
        # }
        
        return preference_data
    
    def train_reward_model(self, preference_dataset: List[Dict]) -> nn.Module:
        """보상 모델 훈련"""
        from transformers import AutoModelForSequenceClassification
        
        # 보상 모델 초기화
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # 선호도 데이터 전처리
        def prepare_preference_data(data):
            prompts = []
            chosen_responses = []
            rejected_responses = []
            
            for item in data:
                prompts.append(item['prompt'])
                chosen_responses.append(item['chosen'])
                rejected_responses.append(item['rejected'])
            
            return prompts, chosen_responses, rejected_responses
        
        prompts, chosen, rejected = prepare_preference_data(preference_dataset)
        
        # 보상 모델 훈련 로직 구현
        # (실제 구현에서는 더 복잡한 로직 필요)
        
        return reward_model
    
    def setup_ppo_trainer(self, reward_model):
        """PPO 트레이너 설정"""
        from trl import PPOTrainer, PPOConfig
        
        # PPO 설정
        ppo_config = PPOConfig(
            model_name=self.base_model_name,
            learning_rate=self.training_stages['rlhf']['learning_rate'],
            batch_size=self.training_stages['rlhf']['batch_size'],
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            ppo_epochs=4,
            seed=42
        )
        
        # PPO 트레이너 초기화
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.peft_model,
            ref_model=None,  # 참조 모델
            tokenizer=self.tokenizer,
            reward_model=reward_model
        )
        
        return ppo_trainer
    
    def compute_qa_metrics(self, eval_pred):
        """Q&A 메트릭 계산"""
        predictions, labels = eval_pred
        
        # 간단한 정확도 계산 (실제로는 더 복잡한 메트릭 필요)
        predictions = np.argmax(predictions, axis=-1)
        
        return {
            'accuracy': accuracy_score(labels.flatten(), predictions.flatten()),
            'f1': f1_score(labels.flatten(), predictions.flatten(), average='weighted')
        }
    
    def compute_instruction_metrics(self, eval_pred):
        """인스트럭션 튜닝 메트릭 계산"""
        predictions, labels = eval_pred
        
        # BLEU, ROUGE 등의 메트릭 계산
        # (실제 구현에서는 더 정교한 메트릭 필요)
        
        return {
            'perplexity': np.exp(np.mean(predictions)),
            'loss': np.mean(predictions)
        }
    
    def load_stage_model(self, stage_name: str):
        """특정 단계의 모델 로드"""
        model_path = f"{self.output_dir}/{stage_name}"
        
        self.peft_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def run_full_pipeline(self, 
                         patent_corpus_path: str,
                         qa_dataset_path: str,
                         instruction_dataset_path: str,
                         preference_dataset_path: str):
        """전체 훈련 파이프라인 실행"""
        
        # 모델 및 토크나이저 설정
        self.setup_model_and_tokenizer()
        
        # 1단계: 연속 사전훈련
        self.stage1_continual_pretraining(patent_corpus_path)
        
        # 2단계: 지도 학습 파인튜닝
        self.stage2_supervised_finetuning(qa_dataset_path)
        
        # 3단계: 인스트럭션 튜닝
        self.stage3_instruction_tuning(instruction_dataset_path)
        
        # 4단계: RLHF
        self.stage4_rlhf_training(preference_dataset_path)
        
        self.logger.info("Full training pipeline completed!")
    
    def evaluate_model(self, test_dataset_path: str) -> Dict:
        """모델 평가"""
        # 테스트 데이터셋 로드
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        
        for item in test_data:
            prompt = item['prompt']
            expected_output = item['expected_output']
            
            # 모델 추론
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'prompt': prompt,
                'expected': expected_output,
                'generated': generated_text,
                'input_length': len(inputs['input_ids'][0]),
                'output_length': len(outputs[0]) - len(inputs['input_ids'][0])
            })
        
        # 평가 메트릭 계산
        evaluation_metrics = self.calculate_evaluation_metrics(results)
        
        return {
            'results': results,
            'metrics': evaluation_metrics
        }
    
    def calculate_evaluation_metrics(self, results: List[Dict]) -> Dict:
        """평가 메트릭 계산"""
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
        
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        
        for result in results:
            expected = result['expected']
            generated = result['generated']
            
            # ROUGE 점수
            rouge_score = rouge_scorer_obj.score(expected, generated)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_score[key].fmeasure)
            
            # BLEU 점수
            bleu_score = sentence_bleu([expected.split()], generated.split())
            bleu_scores.append(bleu_score)
        
        # 평균 계산
        avg_metrics = {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL']),
            'bleu': np.mean(bleu_scores)
        }
        
        return avg_metrics
```

### 13.2 특허 도메인 특화 데이터셋 구축

#### A. 다양한 특허 데이터 소스 통합
```python
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import sqlite3
from typing import Dict, List, Optional
import time
import re
from datetime import datetime, timedelta
import json

class PatentDatasetBuilder:
    """특허 데이터셋 구축을 위한 통합 클래스"""
    
    def __init__(self, db_path: str = "patent_training_data.db"):
        self.db_path = db_path
        self.setup_database()
        
        # API 설정
        self.uspto_api_base = "https://developer.uspto.gov/api/v1"
        self.epo_ops_base = "https://ops.epo.org/3.2/rest-services"
        self.google_patents_base = "https://patents.googleapis.com/v1"
        
        # 요청 제한
        self.request_delay = 1.0  # 초
        
    def setup_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 특허 문서 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patent_number TEXT UNIQUE,
                title TEXT,
                abstract TEXT,
                description TEXT,
                claims TEXT,
                inventors TEXT,
                assignee TEXT,
                filing_date DATE,
                publication_date DATE,
                grant_date DATE,
                classification TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 훈련 데이터 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT,  -- 'continual_pretraining', 'qa', 'instruction', 'preference'
                prompt TEXT,
                response TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 품질 평가 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patent_id INTEGER,
                completeness_score REAL,
                clarity_score REAL,
                technical_depth_score REAL,
                legal_compliance_score REAL,
                overall_score REAL,
                FOREIGN KEY (patent_id) REFERENCES patents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_uspto_patents(self, 
                            query: str, 
                            max_patents: int = 1000,
                            start_date: str = "2020-01-01") -> List[Dict]:
        """USPTO에서 특허 데이터 수집"""
        patents = []
        
        # USPTO API를 통한 검색
        search_url = f"{self.uspto_api_base}/patent/search"
        
        params = {
            'query': query,
            'start': 0,
            'rows': min(100, max_patents),
            'sort': 'date desc'
        }
        
        collected = 0
        while collected < max_patents:
            try:
                response = requests.get(search_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                for patent_data in data.get('patents', []):
                    patent_info = self.parse_uspto_patent(patent_data)
                    if patent_info:
                        patents.append(patent_info)
                        collected += 1
                        
                        if collected >= max_patents:
                            break
                
                # 다음 페이지
                params['start'] += params['rows']
                
                # API 제한 준수
                time.sleep(self.request_delay)
                
            except Exception as e:
                print(f"USPTO API 오류: {e}")
                break
        
        return patents
    
    def collect_epo_patents(self, 
                          query: str, 
                          max_patents: int = 1000) -> List[Dict]:
        """EPO에서 특허 데이터 수집"""
        patents = []
        
        # EPO OPS API를 통한 검색
        search_url = f"{self.epo_ops_base}/published-data/search"
        
        headers = {
            'Accept': 'application/json',
            'X-OPS-Range': f'1-{min(100, max_patents)}'
        }
        
        params = {
            'q': query
        }
        
        try:
            response = requests.get(search_url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            for patent_data in data.get('ops:world-patent-data', {}).get('ops:biblio-search', {}).get('ops:search-result', {}).get('ops:publication-reference', []):
                patent_info = self.parse_epo_patent(patent_data)
                if patent_info:
                    patents.append(patent_info)
            
        except Exception as e:
            print(f"EPO API 오류: {e}")
        
        return patents
    
    def collect_google_patents(self, 
                             query: str, 
                             max_patents: int = 1000) -> List[Dict]:
        """Google Patents에서 특허 데이터 수집"""
        patents = []
        
        # Google Patents API (가상의 엔드포인트)
        # 실제로는 웹 스크래핑이나 다른 방법 필요
        
        # 여기서는 예시 구현
        search_results = self.scrape_google_patents(query, max_patents)
        
        for result in search_results:
            patent_info = self.parse_google_patent(result)
            if patent_info:
                patents.append(patent_info)
        
        return patents
    
    def parse_uspto_patent(self, patent_data: Dict) -> Optional[Dict]:
        """USPTO 특허 데이터 파싱"""
        try:
            return {
                'patent_number': patent_data.get('patentNumber'),
                'title': patent_data.get('title'),
                'abstract': patent_data.get('abstract'),
                'description': patent_data.get('description'),
                'claims': patent_data.get('claims'),
                'inventors': json.dumps(patent_data.get('inventors', [])),
                'assignee': patent_data.get('assignee'),
                'filing_date': patent_data.get('filingDate'),
                'publication_date': patent_data.get('publicationDate'),
                'grant_date': patent_data.get('grantDate'),
                'classification': json.dumps(patent_data.get('classification', [])),
                'source': 'USPTO'
            }
        except Exception as e:
            print(f"USPTO 파싱 오류: {e}")
            return None
    
    def parse_epo_patent(self, patent_data: Dict) -> Optional[Dict]:
        """EPO 특허 데이터 파싱"""
        try:
            # EPO 데이터 구조에 맞게 파싱
            return {
                'patent_number': patent_data.get('document-id', {}).get('doc-number'),
                'title': patent_data.get('title'),
                'abstract': patent_data.get('abstract'),
                'description': patent_data.get('description'),
                'claims': patent_data.get('claims'),
                'inventors': json.dumps(patent_data.get('inventors', [])),
                'assignee': patent_data.get('applicant'),
                'filing_date': patent_data.get('filing-date'),
                'publication_date': patent_data.get('publication-date'),
                'grant_date': patent_data.get('grant-date'),
                'classification': json.dumps(patent_data.get('classification', [])),
                'source': 'EPO'
            }
        except Exception as e:
            print(f"EPO 파싱 오류: {e}")
            return None
    
    def parse_google_patent(self, patent_data: Dict) -> Optional[Dict]:
        """Google Patents 데이터 파싱"""
        try:
            return {
                'patent_number': patent_data.get('patent_number'),
                'title': patent_data.get('title'),
                'abstract': patent_data.get('abstract'),
                'description': patent_data.get('description'),
                'claims': patent_data.get('claims'),
                'inventors': json.dumps(patent_data.get('inventors', [])),
                'assignee': patent_data.get('assignee'),
                'filing_date': patent_data.get('filing_date'),
                'publication_date': patent_data.get('publication_date'),
                'grant_date': patent_data.get('grant_date'),
                'classification': json.dumps(patent_data.get('classification', [])),
                'source': 'Google Patents'
            }
        except Exception as e:
            print(f"Google Patents 파싱 오류: {e}")
            return None
    
    def scrape_google_patents(self, query: str, max_patents: int) -> List[Dict]:
        """Google Patents 웹 스크래핑 (예시)"""
        # 실제 구현에서는 BeautifulSoup 등을 사용
        # 여기서는 더미 데이터 반환
        return []
    
    def save_patents_to_db(self, patents: List[Dict]):
        """특허 데이터를 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for patent in patents:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO patents 
                    (patent_number, title, abstract, description, claims, 
                     inventors, assignee, filing_date, publication_date, 
                     grant_date, classification, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patent['patent_number'],
                    patent['title'],
                    patent['abstract'],
                    patent['description'],
                    patent['claims'],
                    patent['inventors'],
                    patent['assignee'],
                    patent['filing_date'],
                    patent['publication_date'],
                    patent['grant_date'],
                    patent['classification'],
                    patent['source']
                ))
            except Exception as e:
                print(f"데이터베이스 저장 오류: {e}")
        
        conn.commit()
        conn.close()
    
    def generate_continual_pretraining_data(self) -> List[str]:
        """연속 사전훈련용 데이터 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, abstract, description, claims 
            FROM patents 
            WHERE abstract IS NOT NULL AND description IS NOT NULL
        ''')
        
        training_texts = []
        
        for row in cursor.fetchall():
            title, abstract, description, claims = row
            
            # 특허 문서 포맷팅
            patent_text = f"""<patent_start>
제목: {title}

초록: {abstract}

상세 설명: {description}

청구항: {claims}
<patent_end>"""
            
            training_texts.append(patent_text)
        
        conn.close()
        return training_texts
    
    def generate_qa_dataset(self) -> List[Dict]:
        """Q&A 데이터셋 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, abstract, description, claims, classification 
            FROM patents 
            WHERE abstract IS NOT NULL AND claims IS NOT NULL
        ''')
        
        qa_pairs = []
        
        for row in cursor.fetchall():
            title, abstract, description, claims, classification = row
            
            # 다양한 질문 유형 생성
            questions = [
                f"'{title}' 특허의 주요 기술적 특징은 무엇인가요?",
                f"이 특허의 청구항에서 핵심적인 발명의 구성요소는 무엇인가요?",
                f"이 특허가 해결하고자 하는 기술적 문제는 무엇인가요?",
                f"이 특허의 기술 분야는 무엇이며, 어떤 응용 분야에 활용될 수 있나요?",
                f"이 특허의 발명이 기존 기술과 차별화되는 점은 무엇인가요?"
            ]
            
            answers = [
                abstract,
                self.extract_key_elements_from_claims(claims),
                self.extract_technical_problem(description),
                self.extract_application_fields(classification, description),
                self.extract_novelty_aspects(description, claims)
            ]
            
            for question, answer in zip(questions, answers):
                if answer:  # 답변이 있는 경우만 추가
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
        
        conn.close()
        return qa_pairs
    
    def generate_instruction_dataset(self) -> List[Dict]:
        """인스트럭션 데이터셋 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, abstract, description, claims 
            FROM patents 
            WHERE abstract IS NOT NULL AND claims IS NOT NULL
        ''')
        
        instruction_data = []
        
        for row in cursor.fetchall():
            title, abstract, description, claims = row
            
            # 다양한 인스트럭션 태스크
            instructions = [
                {
                    'instruction': '주어진 특허 문서의 초록을 바탕으로 주요 청구항을 생성하세요.',
                    'input': abstract,
                    'output': self.extract_main_claims(claims)
                },
                {
                    'instruction': '특허 제목과 상세 설명을 바탕으로 간결한 초록을 작성하세요.',
                    'input': f"제목: {title}\n\n상세 설명: {description[:1000]}...",
                    'output': abstract
                },
                {
                    'instruction': '특허 청구항을 분석하여 핵심 기술 요소를 추출하세요.',
                    'input': claims,
                    'output': self.extract_key_technical_elements(claims)
                },
                {
                    'instruction': '특허 문서를 바탕으로 발명의 기술적 효과를 설명하세요.',
                    'input': f"제목: {title}\n\n설명: {description[:800]}...",
                    'output': self.extract_technical_effects(description)
                }
            ]
            
            for inst in instructions:
                if inst['output']:  # 출력이 있는 경우만 추가
                    instruction_data.append(inst)
        
        conn.close()
        return instruction_data
    
    def generate_preference_dataset(self) -> List[Dict]:
        """선호도 데이터셋 생성 (RLHF용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 등록된 특허 vs 공개된 특허 비교
        cursor.execute('''
            SELECT title, abstract, claims, grant_date 
            FROM patents 
            WHERE claims IS NOT NULL
        ''')
        
        preference_data = []
        
        for row in cursor.fetchall():
            title, abstract, claims, grant_date = row
            
            prompt = f"다음 특허 초록을 바탕으로 청구항을 작성하세요:\n\n{abstract}"
            
            # 실제 청구항 (선호됨)
            chosen_response = claims
            
            # 품질이 낮은 청구항 생성 (선호되지 않음)
            rejected_response = self.generate_low_quality_claims(abstract)
            
            preference_data.append({
                'prompt': prompt,
                'chosen': chosen_response,
                'rejected': rejected_response,
                'metadata': {
                    'title': title,
                    'is_granted': grant_date is not None
                }
            })
        
        conn.close()
        return preference_data
    
    def extract_key_elements_from_claims(self, claims: str) -> str:
        """청구항에서 핵심 요소 추출"""
        if not claims:
            return ""
        
        # 독립 청구항 추출
        independent_claims = re.findall(r'청구항\s*1[^청구항]*', claims)
        if independent_claims:
            return independent_claims[0].strip()
        
        # 첫 번째 청구항 추출
        first_claim = claims.split('청구항')[1] if '청구항' in claims else claims[:500]
        return first_claim.strip()
    
    def extract_technical_problem(self, description: str) -> str:
        """기술적 문제 추출"""
        if not description:
            return ""
        
        # 문제 관련 키워드 검색
        problem_keywords = ['문제', '과제', '해결', '개선', '단점', '한계']
        
        sentences = description.split('.')
        problem_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in problem_keywords):
                problem_sentences.append(sentence.strip())
        
        return '. '.join(problem_sentences[:3]) if problem_sentences else ""
    
    def extract_application_fields(self, classification: str, description: str) -> str:
        """응용 분야 추출"""
        if not classification and not description:
            return ""
        
        # 분류 정보에서 기술 분야 추출
        if classification:
            try:
                class_data = json.loads(classification)
                if class_data:
                    return f"기술 분류: {', '.join(class_data[:3])}"
            except:
                pass
        
        # 설명에서 응용 분야 키워드 검색
        application_keywords = ['응용', '적용', '사용', '활용', '분야', '산업']
        
        sentences = description.split('.') if description else []
        app_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in application_keywords):
                app_sentences.append(sentence.strip())
        
        return '. '.join(app_sentences[:2]) if app_sentences else ""
    
    def extract_novelty_aspects(self, description: str, claims: str) -> str:
        """신규성 측면 추출"""
        if not description and not claims:
            return ""
        
        novelty_keywords = ['신규', '새로운', '개선된', '향상된', '특징', '차별']
        
        text = (description or "") + " " + (claims or "")
        sentences = text.split('.')
        novelty_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in novelty_keywords):
                novelty_sentences.append(sentence.strip())
        
        return '. '.join(novelty_sentences[:3]) if novelty_sentences else ""
    
    def extract_main_claims(self, claims: str) -> str:
        """주요 청구항 추출"""
        if not claims:
            return ""
        
        # 독립 청구항들 추출
        independent_claims = re.findall(r'청구항\s*\d+[^청구항]*?(?=청구항|\Z)', claims)
        
        # 종속 청구항 제외
        main_claims = []
        for claim in independent_claims:
            if '청구항' in claim and '에 있어서' not in claim and '에 따른' not in claim:
                main_claims.append(claim.strip())
        
        return '\n\n'.join(main_claims[:3]) if main_claims else claims[:1000]
    
    def extract_key_technical_elements(self, claims: str) -> str:
        """핵심 기술 요소 추출"""
        if not claims:
            return ""
        
        # 기술적 구성요소 키워드
        tech_keywords = ['장치', '시스템', '방법', '수단', '모듈', '부', '유닛', '회로', '센서', '프로세서']
        
        sentences = claims.split('.')
        tech_elements = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in tech_keywords):
                tech_elements.append(sentence.strip())
        
        return '. '.join(tech_elements[:5]) if tech_elements else ""
    
    def extract_technical_effects(self, description: str) -> str:
        """기술적 효과 추출"""
        if not description:
            return ""
        
        effect_keywords = ['효과', '장점', '이점', '개선', '향상', '증가', '감소', '최적화']
        
        sentences = description.split('.')
        effect_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in effect_keywords):
                effect_sentences.append(sentence.strip())
        
        return '. '.join(effect_sentences[:4]) if effect_sentences else ""
    
    def generate_low_quality_claims(self, abstract: str) -> str:
        """낮은 품질의 청구항 생성 (RLHF용)"""
        if not abstract:
            return "청구항이 명확하지 않습니다."
        
        # 의도적으로 품질이 낮은 청구항 생성
        low_quality_patterns = [
            f"청구항 1. {abstract[:100]}에 관한 장치.",
            f"청구항 1. 상기 기술을 포함하는 시스템.",
            f"청구항 1. {abstract.split('.')[0]}를 특징으로 하는 방법."
        ]
        
        return low_quality_patterns[0]
    
    def assess_data_quality(self, patent_id: int) -> Dict[str, float]:
        """데이터 품질 평가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, abstract, description, claims 
            FROM patents 
            WHERE id = ?
        ''', (patent_id,))
        
        row = cursor.fetchone()
        if not row:
            return {}
        
        title, abstract, description, claims = row
        
        # 완성도 점수
        completeness_score = self.calculate_completeness_score(title, abstract, description, claims)
        
        # 명확성 점수
        clarity_score = self.calculate_clarity_score(abstract, claims)
        
        # 기술적 깊이 점수
        technical_depth_score = self.calculate_technical_depth_score(description, claims)
        
        # 법적 준수 점수
        legal_compliance_score = self.calculate_legal_compliance_score(claims)
        
        # 전체 점수
        overall_score = (completeness_score + clarity_score + technical_depth_score + legal_compliance_score) / 4
        
        # 품질 점수 저장
        cursor.execute('''
            INSERT OR REPLACE INTO quality_scores 
            (patent_id, completeness_score, clarity_score, technical_depth_score, 
             legal_compliance_score, overall_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (patent_id, completeness_score, clarity_score, technical_depth_score, 
              legal_compliance_score, overall_score))
        
        conn.commit()
        conn.close()
        
        return {
            'completeness_score': completeness_score,
            'clarity_score': clarity_score,
            'technical_depth_score': technical_depth_score,
            'legal_compliance_score': legal_compliance_score,
            'overall_score': overall_score
        }
    
    def calculate_completeness_score(self, title: str, abstract: str, description: str, claims: str) -> float:
        """완성도 점수 계산"""
        score = 0.0
        
        if title and len(title.strip()) > 10:
            score += 0.2
        if abstract and len(abstract.strip()) > 50:
            score += 0.3
        if description and len(description.strip()) > 200:
            score += 0.3
        if claims and len(claims.strip()) > 100:
            score += 0.2
        
        return score
    
    def calculate_clarity_score(self, abstract: str, claims: str) -> float:
        """명확성 점수 계산"""
        score = 0.0
        
        if abstract:
            # 문장 구조 분석
            sentences = abstract.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            if 10 <= avg_sentence_length <= 25:  # 적절한 문장 길이
                score += 0.5
        
        if claims:
            # 청구항 구조 분석
            if '청구항' in claims and '특징으로 하는' in claims:
                score += 0.5
        
        return score
    
    def calculate_technical_depth_score(self, description: str, claims: str) -> float:
        """기술적 깊이 점수 계산"""
        score = 0.0
        
        technical_terms = ['시스템', '방법', '장치', '알고리즘', '프로세스', '모듈', '인터페이스', '프로토콜']
        
        text = (description or "") + " " + (claims or "")
        
        term_count = sum(1 for term in technical_terms if term in text)
        score = min(term_count / len(technical_terms), 1.0)
        
        return score
    
    def calculate_legal_compliance_score(self, claims: str) -> float:
        """법적 준수 점수 계산"""
        score = 0.0
        
        if not claims:
            return score
        
        # 청구항 구조 검사
        if '청구항 1' in claims:
            score += 0.3
        
        # 독립 청구항 존재 검사
        if '특징으로 하는' in claims or '포함하는' in claims:
            score += 0.4
        
        # 종속 청구항 존재 검사
        if '청구항 1에' in claims or '청구항 2에' in claims:
            score += 0.3
        
        return score
    
    def export_training_datasets(self, output_dir: str):
        """훈련 데이터셋 내보내기"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 연속 사전훈련 데이터
        continual_data = self.generate_continual_pretraining_data()
        with open(f"{output_dir}/continual_pretraining.txt", 'w', encoding='utf-8') as f:
            for text in continual_data:
                f.write(text + '\n\n')
        
        # Q&A 데이터
        qa_data = self.generate_qa_dataset()
        with open(f"{output_dir}/qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)
        
        # 인스트럭션 데이터
        instruction_data = self.generate_instruction_dataset()
        with open(f"{output_dir}/instruction_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, ensure_ascii=False, indent=2)
        
        # 선호도 데이터
        preference_data = self.generate_preference_dataset()
        with open(f"{output_dir}/preference_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(preference_data, f, ensure_ascii=False, indent=2)
        
        print(f"훈련 데이터셋이 {output_dir}에 저장되었습니다.")
```

### 13.3 고급 파인튜닝 기법 적용

#### A. 다중 어댑터 및 모듈형 파인튜닝
```python
from peft import AdaLoraConfig, IA3Config, PrefixTuningConfig
import torch.nn.functional as F

class AdvancedPatentFineTuning:
    """고급 특허 LLM 파인튜닝 기법"""
    
    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.adapters = {}
        
    def setup_multi_adapter_system(self):
        """다중 어댑터 시스템 설정"""
        
        # 1. 도메인별 어댑터
        domain_adapters = {
            'mechanical': LoraConfig(
                r=32, lora_alpha=64, 
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1
            ),
            'electrical': LoraConfig(
                r=24, lora_alpha=48,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1
            ),
            'chemical': LoraConfig(
                r=40, lora_alpha=80,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
                lora_dropout=0.05
            ),
            'software': LoraConfig(
                r=48, lora_alpha=96,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05
            )
        }
        
        # 2. 태스크별 어댑터
        task_adapters = {
            'claim_generation': AdaLoraConfig(
                init_r=32, target_r=16, beta1=0.85, beta2=0.85,
                tinit=200, tfinal=1000, deltaT=10,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            'prior_art_search': IA3Config(
                target_modules=["k_proj", "v_proj", "down_proj"],
                feedforward_modules=["down_proj"]
            ),
            'novelty_assessment': PrefixTuningConfig(
                num_virtual_tokens=30,
                prefix_projection=True,
                encoder_hidden_size=768
            ),
            'technical_summary': LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1
            )
        }
        
        # 어댑터 등록
        for domain, config in domain_adapters.items():
            self.base_model.add_adapter(f"domain_{domain}", config)
            
        for task, config in task_adapters.items():
            self.base_model.add_adapter(f"task_{task}", config)
    
    def adaptive_layer_freezing(self, stage: str):
        """적응적 레이어 동결"""
        
        if stage == "early":
            # 초기 단계: 하위 레이어만 훈련
            for name, param in self.base_model.named_parameters():
                if "layers.0." in name or "layers.1." in name or "layers.2." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif stage == "middle":
            # 중간 단계: 중간 레이어 훈련
            for name, param in self.base_model.named_parameters():
                layer_num = self.extract_layer_number(name)
                if layer_num and 3 <= layer_num <= 8:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif stage == "late":
            # 후기 단계: 상위 레이어 훈련
            for name, param in self.base_model.named_parameters():
                layer_num = self.extract_layer_number(name)
                if layer_num and layer_num >= 9:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def extract_layer_number(self, param_name: str) -> Optional[int]:
        """파라미터 이름에서 레이어 번호 추출"""
        import re
        match = re.search(r'layers\.(\d+)\.', param_name)
        return int(match.group(1)) if match else None
    
    def curriculum_learning_schedule(self, epoch: int, total_epochs: int) -> Dict:
        """커리큘럼 학습 스케줄"""
        
        progress = epoch / total_epochs
        
        if progress < 0.3:
            # 초기: 간단한 태스크
            return {
                'task_weights': {
                    'technical_summary': 0.4,
                    'claim_generation': 0.3,
                    'prior_art_search': 0.2,
                    'novelty_assessment': 0.1
                },
                'difficulty_level': 'easy',
                'max_sequence_length': 512
            }
        elif progress < 0.7:
            # 중기: 중간 난이도
            return {
                'task_weights': {
                    'technical_summary': 0.3,
                    'claim_generation': 0.4,
                    'prior_art_search': 0.2,
                    'novelty_assessment': 0.1
                },
                'difficulty_level': 'medium',
                'max_sequence_length': 768
            }
        else:
            # 후기: 복잡한 태스크
            return {
                'task_weights': {
                    'technical_summary': 0.2,
                    'claim_generation': 0.3,
                    'prior_art_search': 0.3,
                    'novelty_assessment': 0.2
                },
                'difficulty_level': 'hard',
                'max_sequence_length': 1024
            }
    
    def knowledge_distillation_training(self, teacher_model, student_model, dataset):
        """지식 증류 훈련"""
        
        def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
            """증류 손실 함수"""
            
            # 소프트 타겟 손실
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
            soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
            
            # 하드 타겟 손실
            hard_loss = F.cross_entropy(student_logits, labels)
            
            # 결합 손실
            return alpha * soft_loss + (1 - alpha) * hard_loss
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-5)
        
        for epoch in range(10):
            for batch in dataset:
                # 교사 모델 추론 (그래디언트 계산 없음)
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                    teacher_logits = teacher_outputs.logits
                
                # 학생 모델 추론
                student_outputs = student_model(**batch)
                student_logits = student_outputs.logits
                
                # 증류 손실 계산
                loss = distillation_loss(
                    student_logits, teacher_logits, batch['labels']
                )
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def meta_learning_adaptation(self, support_set, query_set, num_inner_steps=5):
        """메타 학습 기반 빠른 적응"""
        
        # MAML (Model-Agnostic Meta-Learning) 구현
        meta_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-3)
        
        for episode in range(100):  # 메타 훈련 에피소드
            
            # 내부 루프: 태스크별 적응
            adapted_params = {}
            for name, param in self.base_model.named_parameters():
                adapted_params[name] = param.clone()
            
            # Support set으로 빠른 적응
            for step in range(num_inner_steps):
                support_loss = self.compute_task_loss(support_set, adapted_params)
                
                # 그래디언트 계산
                grads = torch.autograd.grad(
                    support_loss, adapted_params.values(), 
                    create_graph=True, retain_graph=True
                )
                
                # 파라미터 업데이트
                for (name, param), grad in zip(adapted_params.items(), grads):
                    adapted_params[name] = param - 0.01 * grad
            
            # 외부 루프: 메타 업데이트
            query_loss = self.compute_task_loss(query_set, adapted_params)
            
            meta_optimizer.zero_grad()
            query_loss.backward()
            meta_optimizer.step()
    
    def compute_task_loss(self, dataset, params):
        """태스크별 손실 계산"""
        # 실제 구현에서는 더 복잡한 로직 필요
        return torch.tensor(0.0, requires_grad=True)
    
    def progressive_unfreezing(self, total_epochs: int):
        """점진적 언프리징"""
        
        def unfreeze_schedule(epoch: int) -> List[str]:
            """에포크별 언프리징 스케줄"""
            
            progress = epoch / total_epochs
            
            if progress < 0.2:
                return ["embeddings", "layer.0", "layer.1"]
            elif progress < 0.4:
                return ["embeddings", "layer.0", "layer.1", "layer.2", "layer.3"]
            elif progress < 0.6:
                return ["embeddings"] + [f"layer.{i}" for i in range(6)]
            elif progress < 0.8:
                return ["embeddings"] + [f"layer.{i}" for i in range(9)]
            else:
                return ["embeddings"] + [f"layer.{i}" for i in range(12)] + ["pooler"]
        
        return unfreeze_schedule
    
    def domain_adversarial_training(self, source_data, target_data):
        """도메인 적대적 훈련"""
        
        # 도메인 분류기
        domain_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # 소스/타겟 도메인
        )
        
        # 그래디언트 역전 레이어
        class GradientReversalLayer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x.view_as(x)
            
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg() * ctx.alpha, None
        
        def gradient_reversal(x, alpha=1.0):
            return GradientReversalLayer.apply(x, alpha)
        
        # 훈련 루프
        for epoch in range(20):
            for source_batch, target_batch in zip(source_data, target_data):
                
                # 특징 추출
                source_features = self.base_model.get_hidden_states(source_batch)
                target_features = self.base_model.get_hidden_states(target_batch)
                
                # 태스크 손실 (소스 도메인)
                task_loss = self.compute_task_loss(source_batch, None)
                
                # 도메인 적대적 손실
                source_domain_pred = domain_classifier(gradient_reversal(source_features))
                target_domain_pred = domain_classifier(gradient_reversal(target_features))
                
                source_domain_labels = torch.zeros(source_features.size(0), dtype=torch.long)
                target_domain_labels = torch.ones(target_features.size(0), dtype=torch.long)
                
                domain_loss = (
                    F.cross_entropy(source_domain_pred, source_domain_labels) +
                    F.cross_entropy(target_domain_pred, target_domain_labels)
                )
                
                # 총 손실
                total_loss = task_loss + 0.1 * domain_loss
                
                # 역전파
                total_loss.backward()
```

이러한 고도화된 훈련 시스템을 통해 특허 도메인에 특화된 고성능 LLM을 구축할 수 있습니다. 다단계 훈련, 다중 어댑터, 커리큘럼 학습, 지식 증류 등의 최신 기법을 활용하여 모델의 성능과 효율성을 극대화할 수 있습니다.


## 14. 보안, 인증 및 테스트 코드 추가 계획

### 14.1 사용자 인증 및 권한 관리 시스템

#### A. JWT 기반 인증 시스템
```python
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app
import redis
from typing import Dict, Optional, List
import secrets
import re

class PatentLLMAuthSystem:
    """특허 LLM 시스템용 인증 및 권한 관리"""
    
    def __init__(self, app=None, redis_client=None):
        self.app = app
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.secret_key = secrets.token_urlsafe(32)
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Flask 앱 초기화"""
        app.config.setdefault('JWT_SECRET_KEY', self.secret_key)
        app.config.setdefault('JWT_ACCESS_TOKEN_EXPIRES', timedelta(hours=1))
        app.config.setdefault('JWT_REFRESH_TOKEN_EXPIRES', timedelta(days=30))
        
        # 비밀번호 정책
        app.config.setdefault('PASSWORD_MIN_LENGTH', 8)
        app.config.setdefault('PASSWORD_REQUIRE_UPPERCASE', True)
        app.config.setdefault('PASSWORD_REQUIRE_LOWERCASE', True)
        app.config.setdefault('PASSWORD_REQUIRE_NUMBERS', True)
        app.config.setdefault('PASSWORD_REQUIRE_SPECIAL', True)
        
        # 계정 보안 정책
        app.config.setdefault('MAX_LOGIN_ATTEMPTS', 5)
        app.config.setdefault('ACCOUNT_LOCKOUT_DURATION', timedelta(minutes=30))
        app.config.setdefault('SESSION_TIMEOUT', timedelta(hours=2))
    
    def hash_password(self, password: str) -> str:
        """비밀번호 해싱"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """비밀번호 검증"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Dict[str, bool]:
        """비밀번호 강도 검증"""
        validations = {
            'min_length': len(password) >= current_app.config['PASSWORD_MIN_LENGTH'],
            'has_uppercase': bool(re.search(r'[A-Z]', password)) if current_app.config['PASSWORD_REQUIRE_UPPERCASE'] else True,
            'has_lowercase': bool(re.search(r'[a-z]', password)) if current_app.config['PASSWORD_REQUIRE_LOWERCASE'] else True,
            'has_numbers': bool(re.search(r'\d', password)) if current_app.config['PASSWORD_REQUIRE_NUMBERS'] else True,
            'has_special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)) if current_app.config['PASSWORD_REQUIRE_SPECIAL'] else True
        }
        
        validations['is_valid'] = all(validations.values())
        return validations
    
    def generate_tokens(self, user_id: int, user_role: str) -> Dict[str, str]:
        """액세스 토큰 및 리프레시 토큰 생성"""
        now = datetime.utcnow()
        
        # 액세스 토큰
        access_payload = {
            'user_id': user_id,
            'role': user_role,
            'type': 'access',
            'iat': now,
            'exp': now + current_app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }
        
        # 리프레시 토큰
        refresh_payload = {
            'user_id': user_id,
            'type': 'refresh',
            'iat': now,
            'exp': now + current_app.config['JWT_REFRESH_TOKEN_EXPIRES']
        }
        
        access_token = jwt.encode(
            access_payload, 
            current_app.config['JWT_SECRET_KEY'], 
            algorithm='HS256'
        )
        
        refresh_token = jwt.encode(
            refresh_payload, 
            current_app.config['JWT_SECRET_KEY'], 
            algorithm='HS256'
        )
        
        # 리프레시 토큰을 Redis에 저장
        self.redis_client.setex(
            f"refresh_token:{user_id}",
            current_app.config['JWT_REFRESH_TOKEN_EXPIRES'],
            refresh_token
        )
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_in': current_app.config['JWT_ACCESS_TOKEN_EXPIRES'].total_seconds()
        }
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """토큰 검증"""
        try:
            payload = jwt.decode(
                token, 
                current_app.config['JWT_SECRET_KEY'], 
                algorithms=['HS256']
            )
            
            # 토큰이 블랙리스트에 있는지 확인
            if self.is_token_blacklisted(token):
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """액세스 토큰 갱신"""
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get('type') != 'refresh':
            return None
        
        user_id = payload['user_id']
        
        # Redis에서 저장된 리프레시 토큰과 비교
        stored_token = self.redis_client.get(f"refresh_token:{user_id}")
        if not stored_token or stored_token.decode() != refresh_token:
            return None
        
        # 사용자 정보 조회 (실제 구현에서는 데이터베이스에서)
        user_role = self.get_user_role(user_id)
        
        return self.generate_tokens(user_id, user_role)
    
    def blacklist_token(self, token: str):
        """토큰 블랙리스트 추가"""
        payload = self.verify_token(token)
        if payload:
            exp_time = payload['exp']
            current_time = datetime.utcnow().timestamp()
            ttl = int(exp_time - current_time)
            
            if ttl > 0:
                self.redis_client.setex(f"blacklist:{token}", ttl, "1")
    
    def is_token_blacklisted(self, token: str) -> bool:
        """토큰 블랙리스트 확인"""
        return self.redis_client.exists(f"blacklist:{token}")
    
    def track_login_attempt(self, user_id: int, success: bool):
        """로그인 시도 추적"""
        key = f"login_attempts:{user_id}"
        
        if success:
            # 성공 시 시도 횟수 초기화
            self.redis_client.delete(key)
        else:
            # 실패 시 시도 횟수 증가
            attempts = self.redis_client.incr(key)
            if attempts == 1:
                # 첫 번째 실패 시 TTL 설정
                self.redis_client.expire(key, int(current_app.config['ACCOUNT_LOCKOUT_DURATION'].total_seconds()))
            
            # 최대 시도 횟수 초과 시 계정 잠금
            if attempts >= current_app.config['MAX_LOGIN_ATTEMPTS']:
                self.lock_account(user_id)
    
    def is_account_locked(self, user_id: int) -> bool:
        """계정 잠금 상태 확인"""
        return self.redis_client.exists(f"account_locked:{user_id}")
    
    def lock_account(self, user_id: int):
        """계정 잠금"""
        self.redis_client.setex(
            f"account_locked:{user_id}",
            int(current_app.config['ACCOUNT_LOCKOUT_DURATION'].total_seconds()),
            "1"
        )
    
    def unlock_account(self, user_id: int):
        """계정 잠금 해제"""
        self.redis_client.delete(f"account_locked:{user_id}")
        self.redis_client.delete(f"login_attempts:{user_id}")
    
    def get_user_role(self, user_id: int) -> str:
        """사용자 역할 조회 (실제 구현에서는 데이터베이스에서)"""
        # 임시 구현
        return "user"
    
    def require_auth(self, f):
        """인증 필수 데코레이터"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None
            
            # Authorization 헤더에서 토큰 추출
            auth_header = request.headers.get('Authorization')
            if auth_header:
                try:
                    token = auth_header.split(" ")[1]  # "Bearer <token>"
                except IndexError:
                    return jsonify({'error': '잘못된 토큰 형식'}), 401
            
            if not token:
                return jsonify({'error': '토큰이 필요합니다'}), 401
            
            payload = self.verify_token(token)
            if not payload:
                return jsonify({'error': '유효하지 않은 토큰'}), 401
            
            # 계정 잠금 확인
            if self.is_account_locked(payload['user_id']):
                return jsonify({'error': '계정이 잠겨있습니다'}), 423
            
            # 현재 사용자 정보를 request에 추가
            request.current_user = {
                'user_id': payload['user_id'],
                'role': payload['role']
            }
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def require_role(self, required_roles: List[str]):
        """역할 기반 접근 제어 데코레이터"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not hasattr(request, 'current_user'):
                    return jsonify({'error': '인증이 필요합니다'}), 401
                
                user_role = request.current_user.get('role')
                if user_role not in required_roles:
                    return jsonify({'error': '권한이 부족합니다'}), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator

# 사용자 관리 API
class UserManagementAPI:
    """사용자 관리 API"""
    
    def __init__(self, auth_system: PatentLLMAuthSystem, db):
        self.auth = auth_system
        self.db = db
    
    def register_user(self, username: str, email: str, password: str, role: str = 'user') -> Dict:
        """사용자 등록"""
        
        # 비밀번호 강도 검증
        password_validation = self.auth.validate_password_strength(password)
        if not password_validation['is_valid']:
            return {
                'success': False,
                'error': '비밀번호가 보안 요구사항을 충족하지 않습니다',
                'validation': password_validation
            }
        
        # 이메일 중복 확인
        if self.is_email_exists(email):
            return {
                'success': False,
                'error': '이미 등록된 이메일입니다'
            }
        
        # 사용자명 중복 확인
        if self.is_username_exists(username):
            return {
                'success': False,
                'error': '이미 사용 중인 사용자명입니다'
            }
        
        # 비밀번호 해싱
        hashed_password = self.auth.hash_password(password)
        
        # 사용자 생성
        user_id = self.create_user_in_db(username, email, hashed_password, role)
        
        # 이메일 인증 토큰 생성 (선택사항)
        verification_token = self.generate_email_verification_token(user_id)
        
        return {
            'success': True,
            'user_id': user_id,
            'verification_token': verification_token
        }
    
    def login_user(self, email: str, password: str) -> Dict:
        """사용자 로그인"""
        
        # 사용자 조회
        user = self.get_user_by_email(email)
        if not user:
            return {
                'success': False,
                'error': '이메일 또는 비밀번호가 잘못되었습니다'
            }
        
        user_id = user['id']
        
        # 계정 잠금 확인
        if self.auth.is_account_locked(user_id):
            return {
                'success': False,
                'error': '계정이 잠겨있습니다. 나중에 다시 시도해주세요'
            }
        
        # 비밀번호 검증
        if not self.auth.verify_password(password, user['password_hash']):
            self.auth.track_login_attempt(user_id, False)
            return {
                'success': False,
                'error': '이메일 또는 비밀번호가 잘못되었습니다'
            }
        
        # 로그인 성공
        self.auth.track_login_attempt(user_id, True)
        
        # 토큰 생성
        tokens = self.auth.generate_tokens(user_id, user['role'])
        
        # 로그인 기록
        self.log_user_activity(user_id, 'login', request.remote_addr)
        
        return {
            'success': True,
            'tokens': tokens,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        }
    
    def logout_user(self, token: str) -> Dict:
        """사용자 로그아웃"""
        
        # 토큰 블랙리스트 추가
        self.auth.blacklist_token(token)
        
        # 로그아웃 기록
        payload = self.auth.verify_token(token)
        if payload:
            self.log_user_activity(payload['user_id'], 'logout', request.remote_addr)
        
        return {
            'success': True,
            'message': '로그아웃되었습니다'
        }
    
    def change_password(self, user_id: int, current_password: str, new_password: str) -> Dict:
        """비밀번호 변경"""
        
        # 현재 비밀번호 확인
        user = self.get_user_by_id(user_id)
        if not self.auth.verify_password(current_password, user['password_hash']):
            return {
                'success': False,
                'error': '현재 비밀번호가 잘못되었습니다'
            }
        
        # 새 비밀번호 강도 검증
        password_validation = self.auth.validate_password_strength(new_password)
        if not password_validation['is_valid']:
            return {
                'success': False,
                'error': '새 비밀번호가 보안 요구사항을 충족하지 않습니다',
                'validation': password_validation
            }
        
        # 비밀번호 업데이트
        new_password_hash = self.auth.hash_password(new_password)
        self.update_user_password(user_id, new_password_hash)
        
        # 활동 기록
        self.log_user_activity(user_id, 'password_change', request.remote_addr)
        
        return {
            'success': True,
            'message': '비밀번호가 변경되었습니다'
        }
    
    def is_email_exists(self, email: str) -> bool:
        """이메일 중복 확인"""
        # 실제 구현에서는 데이터베이스 쿼리
        return False
    
    def is_username_exists(self, username: str) -> bool:
        """사용자명 중복 확인"""
        # 실제 구현에서는 데이터베이스 쿼리
        return False
    
    def create_user_in_db(self, username: str, email: str, password_hash: str, role: str) -> int:
        """데이터베이스에 사용자 생성"""
        # 실제 구현에서는 데이터베이스 삽입
        return 1
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """이메일로 사용자 조회"""
        # 실제 구현에서는 데이터베이스 쿼리
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """ID로 사용자 조회"""
        # 실제 구현에서는 데이터베이스 쿼리
        return None
    
    def update_user_password(self, user_id: int, password_hash: str):
        """사용자 비밀번호 업데이트"""
        # 실제 구현에서는 데이터베이스 업데이트
        pass
    
    def log_user_activity(self, user_id: int, activity: str, ip_address: str):
        """사용자 활동 기록"""
        # 실제 구현에서는 데이터베이스에 활동 로그 저장
        pass
    
    def generate_email_verification_token(self, user_id: int) -> str:
        """이메일 인증 토큰 생성"""
        payload = {
            'user_id': user_id,
            'type': 'email_verification',
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        
        return jwt.encode(payload, current_app.config['JWT_SECRET_KEY'], algorithm='HS256')
```

#### B. API 키 관리 시스템
```python
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3

class APIKeyManager:
    """API 키 관리 시스템"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """API 키 관리용 데이터베이스 설정"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_name TEXT NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,
                permissions TEXT NOT NULL,  -- JSON 형태
                rate_limit INTEGER DEFAULT 1000,  -- 시간당 요청 수
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used_at TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key_id INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                response_status INTEGER,
                response_time_ms INTEGER,
                request_size INTEGER,
                response_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (api_key_id) REFERENCES api_keys (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_api_key(self, 
                        user_id: int, 
                        key_name: str, 
                        permissions: List[str],
                        rate_limit: int = 1000,
                        expires_in_days: Optional[int] = None) -> Dict:
        """API 키 생성"""
        
        # 32바이트 랜덤 키 생성
        raw_key = secrets.token_bytes(32)
        key_string = secrets.token_urlsafe(32)
        
        # 키 해시 생성 (저장용)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        # 키 프리픽스 (표시용)
        key_prefix = key_string[:8]
        
        # 만료일 설정
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # 데이터베이스에 저장
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_keys 
            (user_id, key_name, key_hash, key_prefix, permissions, rate_limit, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, key_name, key_hash, key_prefix, 
            json.dumps(permissions), rate_limit, expires_at
        ))
        
        api_key_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            'api_key_id': api_key_id,
            'api_key': f"pk_{key_string}",  # 프리픽스 추가
            'key_name': key_name,
            'permissions': permissions,
            'rate_limit': rate_limit,
            'expires_at': expires_at.isoformat() if expires_at else None
        }
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """API 키 검증"""
        
        if not api_key.startswith('pk_'):
            return None
        
        # 프리픽스 제거
        key_string = api_key[3:]
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, user_id, key_name, permissions, rate_limit, is_active, expires_at
            FROM api_keys 
            WHERE key_hash = ? AND is_active = TRUE
        ''', (key_hash,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        api_key_id, user_id, key_name, permissions, rate_limit, is_active, expires_at = row
        
        # 만료 확인
        if expires_at:
            expires_datetime = datetime.fromisoformat(expires_at)
            if datetime.utcnow() > expires_datetime:
                return None
        
        # 사용 기록 업데이트
        self.update_key_usage(api_key_id)
        
        return {
            'api_key_id': api_key_id,
            'user_id': user_id,
            'key_name': key_name,
            'permissions': json.loads(permissions),
            'rate_limit': rate_limit
        }
    
    def update_key_usage(self, api_key_id: int):
        """API 키 사용 기록 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE api_keys 
            SET last_used_at = CURRENT_TIMESTAMP, usage_count = usage_count + 1
            WHERE id = ?
        ''', (api_key_id,))
        
        conn.commit()
        conn.close()
    
    def log_api_usage(self, 
                     api_key_id: int, 
                     endpoint: str, 
                     method: str,
                     ip_address: str,
                     user_agent: str,
                     response_status: int,
                     response_time_ms: int,
                     request_size: int = 0,
                     response_size: int = 0):
        """API 사용 로그 기록"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_usage_logs 
            (api_key_id, endpoint, method, ip_address, user_agent, 
             response_status, response_time_ms, request_size, response_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            api_key_id, endpoint, method, ip_address, user_agent,
            response_status, response_time_ms, request_size, response_size
        ))
        
        conn.commit()
        conn.close()
    
    def check_rate_limit(self, api_key_id: int) -> bool:
        """요청 제한 확인"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 지난 1시간 동안의 요청 수 확인
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        cursor.execute('''
            SELECT COUNT(*) as request_count, ak.rate_limit
            FROM api_usage_logs aul
            JOIN api_keys ak ON aul.api_key_id = ak.id
            WHERE aul.api_key_id = ? AND aul.created_at > ?
            GROUP BY ak.rate_limit
        ''', (api_key_id, one_hour_ago.isoformat()))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return True  # 첫 번째 요청
        
        request_count, rate_limit = row
        return request_count < rate_limit
    
    def revoke_api_key(self, api_key_id: int, user_id: int) -> bool:
        """API 키 비활성화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE api_keys 
            SET is_active = FALSE 
            WHERE id = ? AND user_id = ?
        ''', (api_key_id, user_id))
        
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected_rows > 0
    
    def list_user_api_keys(self, user_id: int) -> List[Dict]:
        """사용자의 API 키 목록 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, key_name, key_prefix, permissions, rate_limit, 
                   is_active, created_at, expires_at, last_used_at, usage_count
            FROM api_keys 
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        api_keys = []
        for row in rows:
            api_keys.append({
                'id': row[0],
                'key_name': row[1],
                'key_prefix': f"pk_{row[2]}***",
                'permissions': json.loads(row[3]),
                'rate_limit': row[4],
                'is_active': bool(row[5]),
                'created_at': row[6],
                'expires_at': row[7],
                'last_used_at': row[8],
                'usage_count': row[9]
            })
        
        return api_keys

# API 키 인증 데코레이터
def require_api_key(permissions: List[str] = None):
    """API 키 인증 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            
            if not api_key:
                return jsonify({'error': 'API 키가 필요합니다'}), 401
            
            # API 키 관리자 인스턴스 (실제 구현에서는 앱 컨텍스트에서 가져옴)
            api_key_manager = current_app.api_key_manager
            
            # API 키 검증
            key_info = api_key_manager.verify_api_key(api_key)
            if not key_info:
                return jsonify({'error': '유효하지 않은 API 키'}), 401
            
            # 요청 제한 확인
            if not api_key_manager.check_rate_limit(key_info['api_key_id']):
                return jsonify({'error': '요청 제한을 초과했습니다'}), 429
            
            # 권한 확인
            if permissions:
                user_permissions = key_info['permissions']
                if not any(perm in user_permissions for perm in permissions):
                    return jsonify({'error': '권한이 부족합니다'}), 403
            
            # 요청 정보를 request에 추가
            request.api_key_info = key_info
            
            # API 사용 로그 기록 (응답 후)
            start_time = datetime.utcnow()
            
            try:
                response = f(*args, **kwargs)
                
                # 응답 시간 계산
                end_time = datetime.utcnow()
                response_time_ms = int((end_time - start_time).total_seconds() * 1000)
                
                # 로그 기록
                api_key_manager.log_api_usage(
                    key_info['api_key_id'],
                    request.endpoint,
                    request.method,
                    request.remote_addr,
                    request.headers.get('User-Agent', ''),
                    response.status_code if hasattr(response, 'status_code') else 200,
                    response_time_ms
                )
                
                return response
                
            except Exception as e:
                # 오류 로그 기록
                end_time = datetime.utcnow()
                response_time_ms = int((end_time - start_time).total_seconds() * 1000)
                
                api_key_manager.log_api_usage(
                    key_info['api_key_id'],
                    request.endpoint,
                    request.method,
                    request.remote_addr,
                    request.headers.get('User-Agent', ''),
                    500,
                    response_time_ms
                )
                
                raise e
        
        return decorated_function
    return decorator
```

### 14.2 보안 강화 조치

#### A. 입력 검증 및 SQL 인젝션 방지
```python
import re
import html
import bleach
from typing import Any, Dict, List, Optional, Union
import sqlparse
from flask import request
import logging

class SecurityValidator:
    """보안 검증 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 허용된 HTML 태그 (매우 제한적)
        self.allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li']
        self.allowed_attributes = {}
        
        # 위험한 패턴들
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"].*['\"])",
            r"(INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>"
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$(){}[\]\\]",
            r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|ping|wget|curl|nc|telnet|ssh|ftp)\b",
            r"(\.\.\/|\.\.\\\\)",
            r"(/etc/passwd|/etc/shadow|/proc/|/sys/)"
        ]
    
    def validate_input(self, 
                      data: Any, 
                      field_name: str, 
                      validation_rules: Dict) -> Dict[str, Any]:
        """입력 데이터 검증"""
        
        result = {
            'is_valid': True,
            'cleaned_data': data,
            'errors': []
        }
        
        if data is None:
            if validation_rules.get('required', False):
                result['is_valid'] = False
                result['errors'].append(f"{field_name}은(는) 필수 항목입니다")
            return result
        
        # 문자열 검증
        if isinstance(data, str):
            result = self._validate_string(data, field_name, validation_rules)
        
        # 숫자 검증
        elif isinstance(data, (int, float)):
            result = self._validate_number(data, field_name, validation_rules)
        
        # 리스트 검증
        elif isinstance(data, list):
            result = self._validate_list(data, field_name, validation_rules)
        
        # 딕셔너리 검증
        elif isinstance(data, dict):
            result = self._validate_dict(data, field_name, validation_rules)
        
        return result
    
    def _validate_string(self, 
                        data: str, 
                        field_name: str, 
                        rules: Dict) -> Dict[str, Any]:
        """문자열 검증"""
        
        result = {
            'is_valid': True,
            'cleaned_data': data,
            'errors': []
        }
        
        # 길이 검증
        if 'min_length' in rules and len(data) < rules['min_length']:
            result['is_valid'] = False
            result['errors'].append(f"{field_name}은(는) 최소 {rules['min_length']}자 이상이어야 합니다")
        
        if 'max_length' in rules and len(data) > rules['max_length']:
            result['is_valid'] = False
            result['errors'].append(f"{field_name}은(는) 최대 {rules['max_length']}자까지 가능합니다")
        
        # 패턴 검증
        if 'pattern' in rules:
            if not re.match(rules['pattern'], data):
                result['is_valid'] = False
                result['errors'].append(f"{field_name}의 형식이 올바르지 않습니다")
        
        # SQL 인젝션 검사
        if rules.get('check_sql_injection', True):
            if self._detect_sql_injection(data):
                result['is_valid'] = False
                result['errors'].append(f"{field_name}에 위험한 SQL 패턴이 감지되었습니다")
                self.logger.warning(f"SQL injection attempt detected in {field_name}: {data}")
        
        # XSS 검사
        if rules.get('check_xss', True):
            if self._detect_xss(data):
                result['is_valid'] = False
                result['errors'].append(f"{field_name}에 위험한 스크립트 패턴이 감지되었습니다")
                self.logger.warning(f"XSS attempt detected in {field_name}: {data}")
        
        # 명령어 인젝션 검사
        if rules.get('check_command_injection', True):
            if self._detect_command_injection(data):
                result['is_valid'] = False
                result['errors'].append(f"{field_name}에 위험한 명령어 패턴이 감지되었습니다")
                self.logger.warning(f"Command injection attempt detected in {field_name}: {data}")
        
        # HTML 정화
        if rules.get('sanitize_html', False):
            result['cleaned_data'] = bleach.clean(
                data, 
                tags=self.allowed_tags, 
                attributes=self.allowed_attributes,
                strip=True
            )
        
        # HTML 이스케이프
        elif rules.get('escape_html', True):
            result['cleaned_data'] = html.escape(data)
        
        return result
    
    def _validate_number(self, 
                        data: Union[int, float], 
                        field_name: str, 
                        rules: Dict) -> Dict[str, Any]:
        """숫자 검증"""
        
        result = {
            'is_valid': True,
            'cleaned_data': data,
            'errors': []
        }
        
        # 범위 검증
        if 'min_value' in rules and data < rules['min_value']:
            result['is_valid'] = False
            result['errors'].append(f"{field_name}은(는) {rules['min_value']} 이상이어야 합니다")
        
        if 'max_value' in rules and data > rules['max_value']:
            result['is_valid'] = False
            result['errors'].append(f"{field_name}은(는) {rules['max_value']} 이하여야 합니다")
        
        # 정수 검증
        if rules.get('integer_only', False) and not isinstance(data, int):
            result['is_valid'] = False
            result['errors'].append(f"{field_name}은(는) 정수여야 합니다")
        
        return result
    
    def _validate_list(self, 
                      data: List, 
                      field_name: str, 
                      rules: Dict) -> Dict[str, Any]:
        """리스트 검증"""
        
        result = {
            'is_valid': True,
            'cleaned_data': data,
            'errors': []
        }
        
        # 길이 검증
        if 'min_items' in rules and len(data) < rules['min_items']:
            result['is_valid'] = False
            result['errors'].append(f"{field_name}은(는) 최소 {rules['min_items']}개 항목이 필요합니다")
        
        if 'max_items' in rules and len(data) > rules['max_items']:
            result['is_valid'] = False
            result['errors'].append(f"{field_name}은(는) 최대 {rules['max_items']}개 항목까지 가능합니다")
        
        # 항목별 검증
        if 'item_rules' in rules:
            cleaned_items = []
            for i, item in enumerate(data):
                item_result = self.validate_input(item, f"{field_name}[{i}]", rules['item_rules'])
                if not item_result['is_valid']:
                    result['is_valid'] = False
                    result['errors'].extend(item_result['errors'])
                cleaned_items.append(item_result['cleaned_data'])
            result['cleaned_data'] = cleaned_items
        
        return result
    
    def _validate_dict(self, 
                      data: Dict, 
                      field_name: str, 
                      rules: Dict) -> Dict[str, Any]:
        """딕셔너리 검증"""
        
        result = {
            'is_valid': True,
            'cleaned_data': {},
            'errors': []
        }
        
        # 필수 키 검증
        if 'required_keys' in rules:
            for key in rules['required_keys']:
                if key not in data:
                    result['is_valid'] = False
                    result['errors'].append(f"{field_name}에 필수 키 '{key}'가 없습니다")
        
        # 허용된 키 검증
        if 'allowed_keys' in rules:
            for key in data.keys():
                if key not in rules['allowed_keys']:
                    result['is_valid'] = False
                    result['errors'].append(f"{field_name}에 허용되지 않은 키 '{key}'가 있습니다")
        
        # 키별 검증
        if 'key_rules' in rules:
            for key, value in data.items():
                if key in rules['key_rules']:
                    key_result = self.validate_input(value, f"{field_name}.{key}", rules['key_rules'][key])
                    if not key_result['is_valid']:
                        result['is_valid'] = False
                        result['errors'].extend(key_result['errors'])
                    result['cleaned_data'][key] = key_result['cleaned_data']
                else:
                    result['cleaned_data'][key] = value
        else:
            result['cleaned_data'] = data
        
        return result
    
    def _detect_sql_injection(self, data: str) -> bool:
        """SQL 인젝션 패턴 감지"""
        data_lower = data.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, data_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_xss(self, data: str) -> bool:
        """XSS 패턴 감지"""
        data_lower = data.lower()
        
        for pattern in self.xss_patterns:
            if re.search(pattern, data_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_command_injection(self, data: str) -> bool:
        """명령어 인젝션 패턴 감지"""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return True
        
        return False
    
    def validate_file_upload(self, file, allowed_extensions: List[str], max_size: int) -> Dict:
        """파일 업로드 검증"""
        
        result = {
            'is_valid': True,
            'errors': []
        }
        
        if not file:
            result['is_valid'] = False
            result['errors'].append("파일이 선택되지 않았습니다")
            return result
        
        # 파일명 검증
        if not file.filename:
            result['is_valid'] = False
            result['errors'].append("파일명이 없습니다")
            return result
        
        # 확장자 검증
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            result['is_valid'] = False
            result['errors'].append(f"허용되지 않은 파일 형식입니다. 허용된 형식: {', '.join(allowed_extensions)}")
        
        # 파일 크기 검증
        file.seek(0, 2)  # 파일 끝으로 이동
        file_size = file.tell()
        file.seek(0)  # 파일 시작으로 복귀
        
        if file_size > max_size:
            result['is_valid'] = False
            result['errors'].append(f"파일 크기가 너무 큽니다. 최대 크기: {max_size // (1024*1024)}MB")
        
        # 파일 내용 검증 (매직 넘버 확인)
        if not self._verify_file_type(file, file_ext):
            result['is_valid'] = False
            result['errors'].append("파일 내용이 확장자와 일치하지 않습니다")
        
        return result
    
    def _verify_file_type(self, file, expected_ext: str) -> bool:
        """파일 타입 검증 (매직 넘버)"""
        
        magic_numbers = {
            'pdf': b'%PDF',
            'jpg': b'\xff\xd8\xff',
            'jpeg': b'\xff\xd8\xff',
            'png': b'\x89PNG\r\n\x1a\n',
            'gif': b'GIF8',
            'txt': None,  # 텍스트 파일은 매직 넘버가 없음
            'json': None,
            'csv': None
        }
        
        if expected_ext not in magic_numbers:
            return False
        
        magic_number = magic_numbers[expected_ext]
        if magic_number is None:
            return True  # 텍스트 파일은 통과
        
        # 파일 시작 부분 읽기
        file.seek(0)
        file_header = file.read(len(magic_number))
        file.seek(0)
        
        return file_header.startswith(magic_number)

# 보안 검증 데코레이터
def validate_request_data(validation_schema: Dict):
    """요청 데이터 검증 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            validator = SecurityValidator()
            
            # JSON 데이터 검증
            if request.is_json:
                data = request.get_json()
                
                for field_name, rules in validation_schema.items():
                    field_data = data.get(field_name)
                    validation_result = validator.validate_input(field_data, field_name, rules)
                    
                    if not validation_result['is_valid']:
                        return jsonify({
                            'error': '입력 데이터 검증 실패',
                            'details': validation_result['errors']
                        }), 400
                    
                    # 정화된 데이터로 교체
                    data[field_name] = validation_result['cleaned_data']
                
                # 정화된 데이터를 request에 저장
                request.validated_data = data
            
            # 폼 데이터 검증
            elif request.form:
                data = request.form.to_dict()
                
                for field_name, rules in validation_schema.items():
                    field_data = data.get(field_name)
                    validation_result = validator.validate_input(field_data, field_name, rules)
                    
                    if not validation_result['is_valid']:
                        return jsonify({
                            'error': '입력 데이터 검증 실패',
                            'details': validation_result['errors']
                        }), 400
                    
                    data[field_name] = validation_result['cleaned_data']
                
                request.validated_data = data
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator
```

### 14.3 종합 테스트 시스템

#### A. 단위 테스트 및 통합 테스트
```python
import unittest
import pytest
import requests
import json
import time
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
from typing import Dict, List, Any
import tempfile
import os
import sqlite3

class PatentLLMTestSuite:
    """특허 LLM 시스템 종합 테스트 스위트"""
    
    def __init__(self, app: Flask, test_db_path: str = None):
        self.app = app
        self.client = app.test_client()
        self.test_db_path = test_db_path or tempfile.mktemp(suffix='.db')
        
        # 테스트용 사용자 데이터
        self.test_users = [
            {
                'username': 'testuser1',
                'email': 'test1@example.com',
                'password': 'TestPass123!',
                'role': 'user'
            },
            {
                'username': 'admin',
                'email': 'admin@example.com',
                'password': 'AdminPass123!',
                'role': 'admin'
            }
        ]
        
        # 테스트용 특허 데이터
        self.test_patents = [
            {
                'title': '인공지능 기반 이미지 인식 시스템',
                'abstract': '본 발명은 딥러닝을 활용한 고정밀 이미지 인식 시스템에 관한 것이다.',
                'claims': '청구항 1. 딥러닝 모델을 포함하는 이미지 인식 시스템.',
                'description': '상세한 기술 설명...'
            }
        ]
    
    def setup_test_environment(self):
        """테스트 환경 설정"""
        # 테스트 데이터베이스 초기화
        self._setup_test_database()
        
        # 테스트 사용자 생성
        self._create_test_users()
        
        # 테스트 데이터 생성
        self._create_test_data()
    
    def teardown_test_environment(self):
        """테스트 환경 정리"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def _setup_test_database(self):
        """테스트 데이터베이스 설정"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 특허 테이블
        cursor.execute('''
            CREATE TABLE patents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                abstract TEXT,
                claims TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 분석 결과 테이블
        cursor.execute('''
            CREATE TABLE analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                query TEXT NOT NULL,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_test_users(self):
        """테스트 사용자 생성"""
        for user_data in self.test_users:
            response = self.client.post('/api/auth/register', 
                                      json=user_data,
                                      content_type='application/json')
            assert response.status_code == 201
    
    def _create_test_data(self):
        """테스트 데이터 생성"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        for patent in self.test_patents:
            cursor.execute('''
                INSERT INTO patents (title, abstract, claims, description)
                VALUES (?, ?, ?, ?)
            ''', (patent['title'], patent['abstract'], patent['claims'], patent['description']))
        
        conn.commit()
        conn.close()

class TestAuthentication(unittest.TestCase):
    """인증 시스템 테스트"""
    
    def setUp(self):
        self.test_suite = PatentLLMTestSuite(app)
        self.test_suite.setup_test_environment()
        self.client = self.test_suite.client
    
    def tearDown(self):
        self.test_suite.teardown_test_environment()
    
    def test_user_registration(self):
        """사용자 등록 테스트"""
        user_data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'NewPass123!',
            'role': 'user'
        }
        
        response = self.client.post('/api/auth/register', 
                                  json=user_data,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('user_id', data)
    
    def test_user_login(self):
        """사용자 로그인 테스트"""
        login_data = {
            'email': 'test1@example.com',
            'password': 'TestPass123!'
        }
        
        response = self.client.post('/api/auth/login',
                                  json=login_data,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('tokens', data)
        self.assertIn('access_token', data['tokens'])
        self.assertIn('refresh_token', data['tokens'])
    
    def test_invalid_login(self):
        """잘못된 로그인 테스트"""
        login_data = {
            'email': 'test1@example.com',
            'password': 'WrongPassword'
        }
        
        response = self.client.post('/api/auth/login',
                                  json=login_data,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 401)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
    
    def test_token_refresh(self):
        """토큰 갱신 테스트"""
        # 먼저 로그인
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'test1@example.com',
                                            'password': 'TestPass123!'
                                        },
                                        content_type='application/json')
        
        login_data = json.loads(login_response.data)
        refresh_token = login_data['tokens']['refresh_token']
        
        # 토큰 갱신
        refresh_response = self.client.post('/api/auth/refresh',
                                          json={'refresh_token': refresh_token},
                                          content_type='application/json')
        
        self.assertEqual(refresh_response.status_code, 200)
        
        refresh_data = json.loads(refresh_response.data)
        self.assertTrue(refresh_data['success'])
        self.assertIn('tokens', refresh_data)
    
    def test_protected_endpoint_without_token(self):
        """토큰 없이 보호된 엔드포인트 접근 테스트"""
        response = self.client.get('/api/patent/search')
        
        self.assertEqual(response.status_code, 401)
    
    def test_protected_endpoint_with_token(self):
        """토큰으로 보호된 엔드포인트 접근 테스트"""
        # 로그인하여 토큰 획득
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'test1@example.com',
                                            'password': 'TestPass123!'
                                        },
                                        content_type='application/json')
        
        login_data = json.loads(login_response.data)
        access_token = login_data['tokens']['access_token']
        
        # 보호된 엔드포인트 접근
        headers = {'Authorization': f'Bearer {access_token}'}
        response = self.client.get('/api/patent/search', headers=headers)
        
        self.assertNotEqual(response.status_code, 401)

class TestPatentAPI(unittest.TestCase):
    """특허 API 테스트"""
    
    def setUp(self):
        self.test_suite = PatentLLMTestSuite(app)
        self.test_suite.setup_test_environment()
        self.client = self.test_suite.client
        
        # 테스트용 토큰 획득
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'test1@example.com',
                                            'password': 'TestPass123!'
                                        },
                                        content_type='application/json')
        
        login_data = json.loads(login_response.data)
        self.access_token = login_data['tokens']['access_token']
        self.headers = {'Authorization': f'Bearer {self.access_token}'}
    
    def tearDown(self):
        self.test_suite.teardown_test_environment()
    
    def test_patent_search(self):
        """특허 검색 테스트"""
        search_data = {
            'query': '인공지능 이미지 인식',
            'implementation_type': 'code'
        }
        
        response = self.client.post('/api/patent/search',
                                  json=search_data,
                                  headers=self.headers,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertIn('analysis', data)
    
    def test_patent_analysis(self):
        """특허 분석 테스트"""
        analysis_data = {
            'patent_id': 1,
            'analysis_type': 'technical_summary'
        }
        
        response = self.client.post('/api/patent/analyze',
                                  json=analysis_data,
                                  headers=self.headers,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('analysis_result', data)
    
    def test_code_generation(self):
        """코드 생성 테스트"""
        generation_data = {
            'patent_description': '딥러닝을 활용한 이미지 분류 시스템',
            'language': 'python',
            'framework': 'tensorflow'
        }
        
        response = self.client.post('/api/patent/generate-code',
                                  json=generation_data,
                                  headers=self.headers,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('generated_code', data)
        self.assertIn('explanation', data)
    
    def test_3d_model_generation(self):
        """3D 모델 생성 테스트"""
        model_data = {
            'patent_figures': ['figure1.png', 'figure2.png'],
            'description': '기계 부품의 3D 구조'
        }
        
        response = self.client.post('/api/patent/generate-3d',
                                  json=model_data,
                                  headers=self.headers,
                                  content_type='application/json')
        
        # 3D 모델 생성은 시간이 오래 걸릴 수 있으므로 202 Accepted 응답 예상
        self.assertIn(response.status_code, [200, 202])
    
    def test_statistics_endpoint(self):
        """통계 엔드포인트 테스트"""
        response = self.client.get('/api/patent/statistics', headers=self.headers)
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('total_patents', data)
        self.assertIn('total_analyses', data)
        self.assertIn('user_activity', data)

class TestSecurity(unittest.TestCase):
    """보안 테스트"""
    
    def setUp(self):
        self.test_suite = PatentLLMTestSuite(app)
        self.test_suite.setup_test_environment()
        self.client = self.test_suite.client
    
    def tearDown(self):
        self.test_suite.teardown_test_environment()
    
    def test_sql_injection_prevention(self):
        """SQL 인젝션 방지 테스트"""
        malicious_data = {
            'query': "'; DROP TABLE users; --",
            'implementation_type': 'code'
        }
        
        # 로그인하여 토큰 획득
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'test1@example.com',
                                            'password': 'TestPass123!'
                                        },
                                        content_type='application/json')
        
        login_data = json.loads(login_response.data)
        headers = {'Authorization': f'Bearer {login_data["tokens"]["access_token"]}'}
        
        response = self.client.post('/api/patent/search',
                                  json=malicious_data,
                                  headers=headers,
                                  content_type='application/json')
        
        # SQL 인젝션이 차단되어야 함
        self.assertEqual(response.status_code, 400)
    
    def test_xss_prevention(self):
        """XSS 방지 테스트"""
        malicious_data = {
            'query': '<script>alert("XSS")</script>',
            'implementation_type': 'code'
        }
        
        # 로그인하여 토큰 획득
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'test1@example.com',
                                            'password': 'TestPass123!'
                                        },
                                        content_type='application/json')
        
        login_data = json.loads(login_response.data)
        headers = {'Authorization': f'Bearer {login_data["tokens"]["access_token"]}'}
        
        response = self.client.post('/api/patent/search',
                                  json=malicious_data,
                                  headers=headers,
                                  content_type='application/json')
        
        # XSS가 차단되어야 함
        self.assertEqual(response.status_code, 400)
    
    def test_rate_limiting(self):
        """요청 제한 테스트"""
        # 로그인하여 토큰 획득
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'test1@example.com',
                                            'password': 'TestPass123!'
                                        },
                                        content_type='application/json')
        
        login_data = json.loads(login_response.data)
        headers = {'Authorization': f'Bearer {login_data["tokens"]["access_token"]}'}
        
        # 빠른 연속 요청
        for i in range(100):
            response = self.client.get('/api/patent/statistics', headers=headers)
            
            # 요청 제한에 걸리면 429 응답
            if response.status_code == 429:
                break
        else:
            self.fail("요청 제한이 작동하지 않습니다")
    
    def test_password_strength_validation(self):
        """비밀번호 강도 검증 테스트"""
        weak_passwords = [
            'password',      # 너무 간단
            '12345678',      # 숫자만
            'abcdefgh',      # 소문자만
            'ABCDEFGH',      # 대문자만
            'Pass123',       # 특수문자 없음
            'P@ss'           # 너무 짧음
        ]
        
        for weak_password in weak_passwords:
            user_data = {
                'username': f'testuser_{weak_password}',
                'email': f'test_{weak_password}@example.com',
                'password': weak_password,
                'role': 'user'
            }
            
            response = self.client.post('/api/auth/register',
                                      json=user_data,
                                      content_type='application/json')
            
            # 약한 비밀번호는 거부되어야 함
            self.assertEqual(response.status_code, 400)

class TestPerformance(unittest.TestCase):
    """성능 테스트"""
    
    def setUp(self):
        self.test_suite = PatentLLMTestSuite(app)
        self.test_suite.setup_test_environment()
        self.client = self.test_suite.client
        
        # 테스트용 토큰 획득
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'test1@example.com',
                                            'password': 'TestPass123!'
                                        },
                                        content_type='application/json')
        
        login_data = json.loads(login_response.data)
        self.headers = {'Authorization': f'Bearer {login_data["tokens"]["access_token"]}'}
    
    def tearDown(self):
        self.test_suite.teardown_test_environment()
    
    def test_response_time(self):
        """응답 시간 테스트"""
        search_data = {
            'query': '인공지능 이미지 인식',
            'implementation_type': 'code'
        }
        
        start_time = time.time()
        
        response = self.client.post('/api/patent/search',
                                  json=search_data,
                                  headers=self.headers,
                                  content_type='application/json')
        
        end_time = time.time()
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 5.0)  # 5초 이내 응답
    
    def test_concurrent_requests(self):
        """동시 요청 테스트"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            search_data = {
                'query': '인공지능 이미지 인식',
                'implementation_type': 'code'
            }
            
            start_time = time.time()
            response = self.client.post('/api/patent/search',
                                      json=search_data,
                                      headers=self.headers,
                                      content_type='application/json')
            end_time = time.time()
            
            results.put({
                'status_code': response.status_code,
                'response_time': end_time - start_time
            })
        
        # 10개의 동시 요청
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        # 결과 검증
        success_count = 0
        total_response_time = 0
        
        while not results.empty():
            result = results.get()
            if result['status_code'] == 200:
                success_count += 1
            total_response_time += result['response_time']
        
        # 최소 80% 성공률
        self.assertGreaterEqual(success_count, 8)
        
        # 평균 응답 시간 10초 이내
        avg_response_time = total_response_time / 10
        self.assertLess(avg_response_time, 10.0)

class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def setUp(self):
        self.test_suite = PatentLLMTestSuite(app)
        self.test_suite.setup_test_environment()
        self.client = self.test_suite.client
    
    def tearDown(self):
        self.test_suite.teardown_test_environment()
    
    def test_full_workflow(self):
        """전체 워크플로우 테스트"""
        
        # 1. 사용자 등록
        user_data = {
            'username': 'integrationtest',
            'email': 'integration@example.com',
            'password': 'IntegrationTest123!',
            'role': 'user'
        }
        
        register_response = self.client.post('/api/auth/register',
                                           json=user_data,
                                           content_type='application/json')
        self.assertEqual(register_response.status_code, 201)
        
        # 2. 로그인
        login_response = self.client.post('/api/auth/login',
                                        json={
                                            'email': 'integration@example.com',
                                            'password': 'IntegrationTest123!'
                                        },
                                        content_type='application/json')
        self.assertEqual(login_response.status_code, 200)
        
        login_data = json.loads(login_response.data)
        headers = {'Authorization': f'Bearer {login_data["tokens"]["access_token"]}'}
        
        # 3. 특허 검색
        search_response = self.client.post('/api/patent/search',
                                         json={
                                             'query': '인공지능 이미지 인식',
                                             'implementation_type': 'code'
                                         },
                                         headers=headers,
                                         content_type='application/json')
        self.assertEqual(search_response.status_code, 200)
        
        # 4. 코드 생성
        code_response = self.client.post('/api/patent/generate-code',
                                       json={
                                           'patent_description': '딥러닝을 활용한 이미지 분류',
                                           'language': 'python',
                                           'framework': 'tensorflow'
                                       },
                                       headers=headers,
                                       content_type='application/json')
        self.assertEqual(code_response.status_code, 200)
        
        # 5. 통계 조회
        stats_response = self.client.get('/api/patent/statistics', headers=headers)
        self.assertEqual(stats_response.status_code, 200)
        
        # 6. 로그아웃
        logout_response = self.client.post('/api/auth/logout',
                                         json={'token': login_data["tokens"]["access_token"]},
                                         headers=headers,
                                         content_type='application/json')
        self.assertEqual(logout_response.status_code, 200)

# 테스트 실행 함수
def run_all_tests():
    """모든 테스트 실행"""
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 클래스들 추가
    test_classes = [
        TestAuthentication,
        TestPatentAPI,
        TestSecurity,
        TestPerformance,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

# pytest 기반 테스트 (추가)
@pytest.fixture
def client():
    """테스트 클라이언트 픽스처"""
    test_suite = PatentLLMTestSuite(app)
    test_suite.setup_test_environment()
    
    yield test_suite.client
    
    test_suite.teardown_test_environment()

@pytest.mark.parametrize("query,expected_status", [
    ("인공지능", 200),
    ("", 400),
    ("a" * 1000, 400),  # 너무 긴 쿼리
])
def test_search_validation(client, query, expected_status):
    """검색 입력 검증 테스트"""
    # 로그인
    login_response = client.post('/api/auth/login',
                               json={
                                   'email': 'test1@example.com',
                                   'password': 'TestPass123!'
                               },
                               content_type='application/json')
    
    login_data = json.loads(login_response.data)
    headers = {'Authorization': f'Bearer {login_data["tokens"]["access_token"]}'}
    
    # 검색 요청
    response = client.post('/api/patent/search',
                         json={
                             'query': query,
                             'implementation_type': 'code'
                         },
                         headers=headers,
                         content_type='application/json')
    
    assert response.status_code == expected_status

if __name__ == '__main__':
    # 테스트 실행
    result = run_all_tests()
    
    if result.wasSuccessful():
        print("\\n모든 테스트가 성공했습니다!")
    else:
        print(f"\\n테스트 실패: {len(result.failures)} 실패, {len(result.errors)} 오류")
```

이러한 종합적인 보안, 인증 및 테스트 시스템을 통해 Patent2Tech LLM 시스템의 안정성과 보안성을 크게 향상시킬 수 있습니다.


## 15. 배포 및 서비스화를 위한 최종 리소스 및 절차 정의

### 15.1 필요한 API 키 및 외부 서비스

#### A. 필수 API 키 목록

**1. OpenAI API 키**
- **용도**: GPT-4/GPT-3.5 모델을 활용한 특허 분석, 코드 생성, 기술 개념 추출
- **요금제**: Pay-as-you-go 또는 월 구독제
- **예상 비용**: 월 $100-500 (사용량에 따라)
- **설정 방법**: 
  ```bash
  export OPENAI_API_KEY="sk-your-api-key-here"
  ```
- **권장 모델**: 
  - GPT-4-turbo: 복잡한 특허 분석용
  - GPT-3.5-turbo: 일반적인 텍스트 처리용
  - text-embedding-ada-002: 벡터 임베딩용

**2. Anthropic Claude API 키 (선택사항)**
- **용도**: 대안 LLM으로 활용, 특히 긴 문서 처리에 유리
- **요금제**: Pay-as-you-go
- **예상 비용**: 월 $50-200
- **설정 방법**:
  ```bash
  export ANTHROPIC_API_KEY="sk-ant-your-api-key-here"
  ```

**3. Google Cloud Platform API 키**
- **용도**: 
  - Google Patents API 접근
  - Cloud Vision API (특허 도면 OCR)
  - Cloud Translation API (다국어 특허 번역)
  - Cloud Storage (파일 저장)
- **예상 비용**: 월 $50-200
- **필요한 서비스**:
  - Patents API
  - Vision API
  - Translation API
  - Cloud Storage
  - Compute Engine (배포용)

**4. USPTO API 접근**
- **용도**: 미국 특허 데이터 수집
- **비용**: 무료 (API 제한 있음)
- **API 키**: 필요 없음 (공개 API)
- **제한사항**: 시간당 요청 수 제한

**5. EPO Open Patent Services (OPS) API**
- **용도**: 유럽 특허 데이터 수집
- **비용**: 무료 (등록 필요)
- **API 키**: 필요
- **등록 절차**: EPO 개발자 포털에서 신청

**6. KIPO API (한국특허정보원)**
- **용도**: 한국 특허 데이터 수집
- **비용**: 무료/유료 (사용량에 따라)
- **API 키**: 필요
- **등록 절차**: KIPRIS 개발자 센터에서 신청

**7. Hugging Face API 키**
- **용도**: 
  - 사전 훈련된 특허 관련 모델 사용
  - 모델 호스팅 및 추론
- **비용**: 무료 티어 + 유료 Pro ($9/월)
- **설정 방법**:
  ```bash
  export HUGGINGFACE_API_KEY="hf_your-api-key-here"
  ```

**8. Redis Cloud 또는 AWS ElastiCache**
- **용도**: 세션 관리, 캐싱, 요청 제한
- **비용**: 월 $15-100 (용량에 따라)
- **대안**: 자체 호스팅 Redis 서버

**9. 이메일 서비스 API**
- **SendGrid**: 이메일 인증, 알림 발송
- **비용**: 월 $15-50
- **설정 방법**:
  ```bash
  export SENDGRID_API_KEY="SG.your-api-key-here"
  ```

#### B. 3D 모델링 관련 서비스

**1. NeRF 구현을 위한 GPU 리소스**
- **AWS EC2 P3/P4 인스턴스**: NVIDIA V100/A100 GPU
- **Google Cloud Compute Engine**: NVIDIA T4/V100 GPU
- **Azure NC 시리즈**: NVIDIA K80/V100 GPU
- **예상 비용**: 시간당 $1-8 (GPU 타입에 따라)

**2. 3D 모델 저장 및 렌더링**
- **AWS S3**: 3D 모델 파일 저장
- **CloudFront**: 3D 모델 CDN 배포
- **Three.js/Babylon.js**: 웹 3D 렌더링

### 15.2 인프라 및 배포 환경

#### A. 클라우드 인프라 구성

**1. 프로덕션 환경 아키텍처**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Servers   │    │   API Servers   │
│   (ALB/NLB)     │────│   (React App)   │────│   (Flask API)   │
│                 │    │   Auto Scaling  │    │   Auto Scaling  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudFront    │    │   S3 Bucket     │    │   RDS Database  │
│   (CDN)         │    │   (Static Files)│    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ElastiCache   │    │   GPU Instances │    │   Monitoring    │
│   (Redis)       │    │   (NeRF/3D)     │    │   (CloudWatch)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**2. AWS 기반 배포 구성**

```yaml
# docker-compose.prod.yml
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
      - DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/patent_db
      - REDIS_URL=redis://elasticache-endpoint:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
      - SENDGRID_API_KEY=${SENDGRID_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

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

  worker:
    build:
      context: ./patent_llm_api
      dockerfile: Dockerfile.worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://user:pass@postgres:5432/patent_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - postgres

volumes:
  redis_data:
  postgres_data:
```

**3. Kubernetes 배포 구성 (선택사항)**

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: patent-llm-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: patent-llm-backend
  template:
    metadata:
      labels:
        app: patent-llm-backend
    spec:
      containers:
      - name: backend
        image: patent2tech/backend:latest
        ports:
        - containerPort: 5000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: patent-llm-backend-service
spec:
  selector:
    app: patent-llm-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

#### B. 데이터베이스 설계 및 마이그레이션

**1. PostgreSQL 스키마 설계**

```sql
-- 사용자 테이블
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    login_count INTEGER DEFAULT 0
);

-- 특허 테이블
CREATE TABLE patents (
    id SERIAL PRIMARY KEY,
    patent_number VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    claims TEXT,
    description TEXT,
    inventors TEXT[],
    assignees TEXT[],
    filing_date DATE,
    publication_date DATE,
    grant_date DATE,
    patent_office VARCHAR(10), -- USPTO, EPO, KIPO, etc.
    classification_codes TEXT[],
    citations INTEGER[],
    figures_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 특허 도면 테이블
CREATE TABLE patent_figures (
    id SERIAL PRIMARY KEY,
    patent_id INTEGER REFERENCES patents(id),
    figure_number VARCHAR(20),
    figure_type VARCHAR(50), -- diagram, flowchart, schematic, etc.
    file_path TEXT,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    ocr_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 분석 결과 테이블
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    patent_id INTEGER REFERENCES patents(id),
    query TEXT NOT NULL,
    analysis_type VARCHAR(50), -- search, code_generation, 3d_modeling, etc.
    result JSONB,
    processing_time_ms INTEGER,
    tokens_used INTEGER,
    cost_usd DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 생성된 코드 테이블
CREATE TABLE generated_codes (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analysis_results(id),
    language VARCHAR(20),
    framework VARCHAR(50),
    code_content TEXT,
    explanation TEXT,
    test_cases TEXT,
    dependencies TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3D 모델 테이블
CREATE TABLE generated_3d_models (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analysis_results(id),
    model_type VARCHAR(50), -- nerf, mesh, point_cloud, etc.
    file_path TEXT,
    file_size INTEGER,
    processing_time_ms INTEGER,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API 키 테이블
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    key_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(20) NOT NULL,
    permissions JSONB NOT NULL,
    rate_limit INTEGER DEFAULT 1000,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    usage_count INTEGER DEFAULT 0
);

-- API 사용 로그 테이블
CREATE TABLE api_usage_logs (
    id SERIAL PRIMARY KEY,
    api_key_id INTEGER REFERENCES api_keys(id),
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    request_size INTEGER,
    response_size INTEGER,
    response_status INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 활동 로그 테이블
CREATE TABLE user_activity_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    activity_type VARCHAR(50) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 시스템 설정 테이블
CREATE TABLE system_settings (
    id SERIAL PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB,
    description TEXT,
    updated_by INTEGER REFERENCES users(id),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX idx_patents_patent_number ON patents(patent_number);
CREATE INDEX idx_patents_title ON patents USING gin(to_tsvector('english', title));
CREATE INDEX idx_patents_abstract ON patents USING gin(to_tsvector('english', abstract));
CREATE INDEX idx_patents_filing_date ON patents(filing_date);
CREATE INDEX idx_patents_patent_office ON patents(patent_office);

CREATE INDEX idx_analysis_results_user_id ON analysis_results(user_id);
CREATE INDEX idx_analysis_results_created_at ON analysis_results(created_at);
CREATE INDEX idx_analysis_results_analysis_type ON analysis_results(analysis_type);

CREATE INDEX idx_api_usage_logs_api_key_id ON api_usage_logs(api_key_id);
CREATE INDEX idx_api_usage_logs_created_at ON api_usage_logs(created_at);

CREATE INDEX idx_user_activity_logs_user_id ON user_activity_logs(user_id);
CREATE INDEX idx_user_activity_logs_created_at ON user_activity_logs(created_at);
```

**2. 데이터베이스 마이그레이션 스크립트**

```python
# migrations/001_initial_schema.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # 사용자 테이블 생성
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('email_verified', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('last_login_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('login_count', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    
    # 특허 테이블 생성
    op.create_table('patents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patent_number', sa.String(length=50), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('claims', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('inventors', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('assignees', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('filing_date', sa.Date(), nullable=True),
        sa.Column('publication_date', sa.Date(), nullable=True),
        sa.Column('grant_date', sa.Date(), nullable=True),
        sa.Column('patent_office', sa.String(length=10), nullable=True),
        sa.Column('classification_codes', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('citations', postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column('figures_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('patent_number')
    )
    
    # 나머지 테이블들...
    
    # 인덱스 생성
    op.create_index('idx_patents_patent_number', 'patents', ['patent_number'])
    op.create_index('idx_patents_filing_date', 'patents', ['filing_date'])
    op.create_index('idx_patents_patent_office', 'patents', ['patent_office'])

def downgrade():
    op.drop_index('idx_patents_patent_office')
    op.drop_index('idx_patents_filing_date')
    op.drop_index('idx_patents_patent_number')
    op.drop_table('patents')
    op.drop_table('users')
```

### 15.3 배포 자동화 및 CI/CD 파이프라인

#### A. GitHub Actions 워크플로우

```yaml
# .github/workflows/deploy.yml
name: Deploy Patent2Tech LLM System

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install Python dependencies
      run: |
        cd patent_llm_api
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Install Node.js dependencies
      run: |
        cd patent_llm_frontend
        npm ci
    
    - name: Run Python tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        cd patent_llm_api
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Run Frontend tests
      run: |
        cd patent_llm_frontend
        npm test -- --coverage --watchAll=false
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./patent_llm_api/coverage.xml,./patent_llm_frontend/coverage/lcov.info

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata (tags, labels) for Backend
      id: meta-backend
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-backend

    - name: Build and push Backend Docker image
      uses: docker/build-push-action@v4
      with:
        context: ./patent_llm_api
        push: true
        tags: ${{ steps.meta-backend.outputs.tags }}
        labels: ${{ steps.meta-backend.outputs.labels }}

    - name: Extract metadata (tags, labels) for Frontend
      id: meta-frontend
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-frontend

    - name: Build and push Frontend Docker image
      uses: docker/build-push-action@v4
      with:
        context: ./patent_llm_frontend
        push: true
        tags: ${{ steps.meta-frontend.outputs.tags }}
        labels: ${{ steps.meta-frontend.outputs.labels }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /opt/patent2tech
          docker-compose pull
          docker-compose up -d --remove-orphans
          docker system prune -f
```

#### B. Docker 컨테이너 구성

**1. Backend Dockerfile**

```dockerfile
# patent_llm_api/Dockerfile.prod
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 비루트 사용자 생성
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 환경 변수 설정
ENV FLASK_APP=src/main.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "src.main:app"]
```

**2. Frontend Dockerfile**

```dockerfile
# patent_llm_frontend/Dockerfile.prod
# 빌드 스테이지
FROM node:18-alpine as build

WORKDIR /app

# 의존성 설치
COPY package*.json ./
RUN npm ci --only=production

# 소스 코드 복사 및 빌드
COPY . .
RUN npm run build

# 프로덕션 스테이지
FROM nginx:alpine

# 빌드된 파일 복사
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx 설정 복사
COPY nginx.conf /etc/nginx/nginx.conf

# 비루트 사용자로 실행
RUN addgroup -g 1001 -S nodejs \
    && adduser -S nextjs -u 1001 \
    && chown -R nextjs:nodejs /usr/share/nginx/html \
    && chown -R nextjs:nodejs /var/cache/nginx \
    && chown -R nextjs:nodejs /var/log/nginx \
    && chown -R nextjs:nodejs /etc/nginx/conf.d

USER nextjs

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**3. Nginx 설정**

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # 로그 형식
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # 기본 설정
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip 압축
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # 업스트림 서버 정의
    upstream backend {
        server backend:5000;
    }

    # 메인 서버 블록
    server {
        listen 80;
        server_name patent2tech.com www.patent2tech.com;

        # HTTPS로 리다이렉트
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name patent2tech.com www.patent2tech.com;

        # SSL 설정
        ssl_certificate /etc/ssl/certs/patent2tech.crt;
        ssl_certificate_key /etc/ssl/private/patent2tech.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # 보안 헤더
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

        # 정적 파일 서빙
        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files $uri $uri/ /index.html;

            # 캐시 설정
            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
                expires 1y;
                add_header Cache-Control "public, immutable";
            }
        }

        # API 프록시
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 타임아웃 설정
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # 버퍼링 설정
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }

        # WebSocket 지원 (필요시)
        location /ws/ {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # 파일 업로드 크기 제한
        client_max_body_size 100M;

        # 헬스체크 엔드포인트
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

### 15.4 모니터링 및 로깅 시스템

#### A. 애플리케이션 모니터링

**1. Prometheus + Grafana 설정**

```yaml
# monitoring/docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'

volumes:
  prometheus_data:
  grafana_data:
```

**2. 애플리케이션 메트릭 수집**

```python
# src/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Response
import time
import psutil
import threading

class PatentLLMMetrics:
    """특허 LLM 시스템 메트릭 수집"""
    
    def __init__(self):
        # API 요청 메트릭
        self.api_requests_total = Counter(
            'patent_llm_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.api_request_duration = Histogram(
            'patent_llm_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint']
        )
        
        # LLM 관련 메트릭
        self.llm_requests_total = Counter(
            'patent_llm_llm_requests_total',
            'Total LLM requests',
            ['model', 'task_type']
        )
        
        self.llm_tokens_used = Counter(
            'patent_llm_tokens_used_total',
            'Total tokens used',
            ['model', 'token_type']
        )
        
        self.llm_cost_usd = Counter(
            'patent_llm_cost_usd_total',
            'Total cost in USD',
            ['model']
        )
        
        # 시스템 메트릭
        self.active_users = Gauge(
            'patent_llm_active_users',
            'Number of active users'
        )
        
        self.database_connections = Gauge(
            'patent_llm_database_connections',
            'Number of database connections'
        )
        
        self.memory_usage = Gauge(
            'patent_llm_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'patent_llm_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # 비즈니스 메트릭
        self.patents_processed = Counter(
            'patent_llm_patents_processed_total',
            'Total patents processed'
        )
        
        self.code_generated = Counter(
            'patent_llm_code_generated_total',
            'Total code generations',
            ['language', 'framework']
        )
        
        self.models_3d_generated = Counter(
            'patent_llm_3d_models_generated_total',
            'Total 3D models generated'
        )
        
        # 시스템 메트릭 수집 시작
        self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """시스템 메트릭 수집 스레드 시작"""
        def collect_system_metrics():
            while True:
                try:
                    # 메모리 사용량
                    memory = psutil.virtual_memory()
                    self.memory_usage.set(memory.used)
                    
                    # CPU 사용량
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)
                    
                    time.sleep(30)  # 30초마다 수집
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_api_request(self, method: str, endpoint: str, status: int, duration: float):
        """API 요청 메트릭 기록"""
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_llm_usage(self, model: str, task_type: str, tokens_used: int, cost_usd: float):
        """LLM 사용 메트릭 기록"""
        self.llm_requests_total.labels(
            model=model,
            task_type=task_type
        ).inc()
        
        self.llm_tokens_used.labels(
            model=model,
            token_type='total'
        ).inc(tokens_used)
        
        self.llm_cost_usd.labels(model=model).inc(cost_usd)
    
    def record_business_metric(self, metric_type: str, **labels):
        """비즈니스 메트릭 기록"""
        if metric_type == 'patent_processed':
            self.patents_processed.inc()
        elif metric_type == 'code_generated':
            self.code_generated.labels(
                language=labels.get('language', 'unknown'),
                framework=labels.get('framework', 'unknown')
            ).inc()
        elif metric_type == '3d_model_generated':
            self.models_3d_generated.inc()
    
    def get_metrics(self):
        """메트릭 데이터 반환"""
        return generate_latest()

# Flask 앱에 메트릭 통합
def setup_metrics(app):
    """Flask 앱에 메트릭 설정"""
    metrics = PatentLLMMetrics()
    
    @app.before_request
    def before_request():
        request.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            metrics.record_api_request(
                method=request.method,
                endpoint=request.endpoint or 'unknown',
                status=response.status_code,
                duration=duration
            )
        return response
    
    @app.route('/metrics')
    def metrics_endpoint():
        return Response(metrics.get_metrics(), mimetype='text/plain')
    
    app.metrics = metrics
    return metrics
```

#### B. 로깅 시스템

**1. 구조화된 로깅 설정**

```python
# src/logging_config.py
import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """JSON 형태로 로그를 포맷팅"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 추가 필드가 있으면 포함
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'ip_address'):
            log_entry['ip_address'] = record.ip_address
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging(app):
    """로깅 설정"""
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': JSONFormatter
            },
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'json' if app.config.get('FLASK_ENV') == 'production' else 'standard',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json',
                'filename': 'logs/patent_llm.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': 'logs/patent_llm_errors.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # 루트 로거
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'patent_llm': {
                'handlers': ['console', 'file', 'error_file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'werkzeug': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Flask 앱 로거 설정
    app.logger.setLevel(logging.INFO)
    
    return logging.getLogger('patent_llm')

class RequestLogger:
    """요청 로깅 미들웨어"""
    
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger('patent_llm.requests')
    
    def __call__(self, environ, start_response):
        def new_start_response(status, response_headers):
            # 응답 로깅
            self.logger.info(
                "Request completed",
                extra={
                    'method': environ.get('REQUEST_METHOD'),
                    'path': environ.get('PATH_INFO'),
                    'status': status,
                    'ip_address': environ.get('REMOTE_ADDR'),
                    'user_agent': environ.get('HTTP_USER_AGENT')
                }
            )
            return start_response(status, response_headers)
        
        # 요청 로깅
        self.logger.info(
            "Request started",
            extra={
                'method': environ.get('REQUEST_METHOD'),
                'path': environ.get('PATH_INFO'),
                'ip_address': environ.get('REMOTE_ADDR'),
                'user_agent': environ.get('HTTP_USER_AGENT')
            }
        )
        
        return self.app(environ, new_start_response)

# 보안 이벤트 로깅
class SecurityLogger:
    """보안 관련 이벤트 로깅"""
    
    def __init__(self):
        self.logger = logging.getLogger('patent_llm.security')
    
    def log_login_attempt(self, email: str, success: bool, ip_address: str):
        """로그인 시도 로깅"""
        self.logger.info(
            f"Login attempt: {'success' if success else 'failed'}",
            extra={
                'event_type': 'login_attempt',
                'email': email,
                'success': success,
                'ip_address': ip_address
            }
        )
    
    def log_suspicious_activity(self, activity_type: str, details: Dict[str, Any], ip_address: str):
        """의심스러운 활동 로깅"""
        self.logger.warning(
            f"Suspicious activity detected: {activity_type}",
            extra={
                'event_type': 'suspicious_activity',
                'activity_type': activity_type,
                'details': details,
                'ip_address': ip_address
            }
        )
    
    def log_api_abuse(self, api_key_id: int, abuse_type: str, ip_address: str):
        """API 남용 로깅"""
        self.logger.warning(
            f"API abuse detected: {abuse_type}",
            extra={
                'event_type': 'api_abuse',
                'api_key_id': api_key_id,
                'abuse_type': abuse_type,
                'ip_address': ip_address
            }
        )
```

### 15.5 비용 추정 및 운영 계획

#### A. 월간 운영 비용 추정

**1. 클라우드 인프라 비용 (AWS 기준)**

| 서비스 | 사양 | 월 비용 (USD) | 설명 |
|--------|------|---------------|------|
| EC2 (웹서버) | t3.medium × 2 | $60 | 로드밸런싱된 웹서버 |
| EC2 (API서버) | t3.large × 2 | $120 | API 처리 서버 |
| EC2 (GPU) | p3.2xlarge | $918 | NeRF/3D 모델링용 (필요시) |
| RDS PostgreSQL | db.t3.medium | $65 | 관리형 데이터베이스 |
| ElastiCache Redis | cache.t3.micro | $15 | 세션/캐시 저장소 |
| S3 Storage | 100GB | $23 | 파일 저장 |
| CloudFront CDN | 1TB 전송 | $85 | 콘텐츠 배포 |
| Application Load Balancer | 1개 | $23 | 로드밸런서 |
| **소계** | | **$1,309** | GPU 포함 시 |
| **소계 (GPU 제외)** | | **$391** | 일반 운영 시 |

**2. 외부 API 비용**

| 서비스 | 사용량 | 월 비용 (USD) | 설명 |
|--------|--------|---------------|------|
| OpenAI API | 1M 토큰 | $20-60 | GPT-4/3.5 사용 |
| Google Cloud APIs | 10K 요청 | $50 | Vision, Translation 등 |
| SendGrid | 10K 이메일 | $15 | 이메일 발송 |
| **소계** | | **$85-125** | |

**3. 총 운영 비용 추정**

- **최소 구성 (GPU 없음)**: $476-516/월
- **완전 구성 (GPU 포함)**: $1,394-1,434/월
- **연간 비용**: $5,712-17,208

#### B. 확장성 계획

**1. 사용자 증가에 따른 확장 단계**

| 단계 | 사용자 수 | 인프라 구성 | 월 비용 (USD) |
|------|-----------|-------------|---------------|
| 스타트업 | 100-500 | 기본 구성 | $500 |
| 성장기 | 500-5,000 | 중간 구성 | $1,500 |
| 확장기 | 5,000-50,000 | 대규모 구성 | $5,000 |
| 엔터프라이즈 | 50,000+ | 멀티 리전 | $15,000+ |

**2. 자동 확장 설정**

```yaml
# auto-scaling.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: patent-llm-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: patent-llm-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 15.6 보안 및 컴플라이언스

#### A. 데이터 보호 및 개인정보 처리

**1. GDPR/CCPA 준수**

```python
# src/privacy.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import json

class PrivacyManager:
    """개인정보 보호 관리"""
    
    def __init__(self, db):
        self.db = db
    
    def anonymize_user_data(self, user_id: int) -> Dict:
        """사용자 데이터 익명화"""
        
        # 사용자 정보 익명화
        anonymous_id = hashlib.sha256(f"user_{user_id}_{datetime.utcnow()}".encode()).hexdigest()[:16]
        
        # 개인 식별 정보 제거
        self.db.execute("""
            UPDATE users 
            SET 
                username = %s,
                email = %s,
                is_active = FALSE,
                anonymized_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (f"anonymous_{anonymous_id}", f"anonymous_{anonymous_id}@deleted.com", user_id))
        
        # 분석 결과에서 개인 정보 제거
        self.db.execute("""
            UPDATE analysis_results 
            SET query = 'ANONYMIZED'
            WHERE user_id = %s AND query LIKE '%personal%'
        """, (user_id,))
        
        return {
            'user_id': user_id,
            'anonymous_id': anonymous_id,
            'anonymized_at': datetime.utcnow().isoformat()
        }
    
    def export_user_data(self, user_id: int) -> Dict:
        """사용자 데이터 내보내기 (GDPR 요청)"""
        
        # 사용자 기본 정보
        user_data = self.db.fetchone("""
            SELECT username, email, created_at, last_login_at
            FROM users WHERE id = %s
        """, (user_id,))
        
        # 분석 기록
        analysis_data = self.db.fetchall("""
            SELECT query, analysis_type, created_at
            FROM analysis_results WHERE user_id = %s
        """, (user_id,))
        
        # API 사용 기록
        api_usage = self.db.fetchall("""
            SELECT ak.key_name, aul.endpoint, aul.created_at
            FROM api_usage_logs aul
            JOIN api_keys ak ON aul.api_key_id = ak.id
            WHERE ak.user_id = %s
        """, (user_id,))
        
        return {
            'user_info': user_data,
            'analysis_history': analysis_data,
            'api_usage': api_usage,
            'exported_at': datetime.utcnow().isoformat()
        }
    
    def delete_user_data(self, user_id: int) -> Dict:
        """사용자 데이터 완전 삭제"""
        
        # 관련 데이터 삭제 순서 (외래키 제약 고려)
        tables_to_clean = [
            'api_usage_logs',
            'api_keys',
            'generated_3d_models',
            'generated_codes',
            'analysis_results',
            'user_activity_logs',
            'users'
        ]
        
        deleted_records = {}
        
        for table in tables_to_clean:
            if table == 'api_usage_logs':
                count = self.db.execute("""
                    DELETE FROM api_usage_logs 
                    WHERE api_key_id IN (
                        SELECT id FROM api_keys WHERE user_id = %s
                    )
                """, (user_id,))
            elif table == 'users':
                count = self.db.execute("""
                    DELETE FROM users WHERE id = %s
                """, (user_id,))
            else:
                count = self.db.execute(f"""
                    DELETE FROM {table} WHERE user_id = %s
                """, (user_id,))
            
            deleted_records[table] = count
        
        return {
            'user_id': user_id,
            'deleted_records': deleted_records,
            'deleted_at': datetime.utcnow().isoformat()
        }
    
    def schedule_data_retention_cleanup(self):
        """데이터 보존 정책에 따른 정리"""
        
        # 90일 이상 된 로그 삭제
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        
        self.db.execute("""
            DELETE FROM api_usage_logs 
            WHERE created_at < %s
        """, (cutoff_date,))
        
        self.db.execute("""
            DELETE FROM user_activity_logs 
            WHERE created_at < %s
        """, (cutoff_date,))
        
        # 1년 이상 비활성 사용자 알림
        inactive_cutoff = datetime.utcnow() - timedelta(days=365)
        
        inactive_users = self.db.fetchall("""
            SELECT id, email FROM users 
            WHERE last_login_at < %s AND is_active = TRUE
        """, (inactive_cutoff,))
        
        return {
            'logs_cleaned': True,
            'inactive_users_count': len(inactive_users),
            'cleanup_date': datetime.utcnow().isoformat()
        }
```

**2. 보안 감사 및 취약점 스캔**

```bash
#!/bin/bash
# security_audit.sh

echo "Starting security audit..."

# 1. 의존성 취약점 스캔
echo "Scanning Python dependencies..."
cd patent_llm_api
pip-audit --format=json --output=security_audit_python.json

echo "Scanning Node.js dependencies..."
cd ../patent_llm_frontend
npm audit --json > security_audit_nodejs.json

# 2. 코드 보안 스캔
echo "Running static code analysis..."
cd ../patent_llm_api
bandit -r src/ -f json -o security_audit_bandit.json

# 3. Docker 이미지 스캔
echo "Scanning Docker images..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/tmp aquasec/trivy image patent2tech/backend:latest \
  --format json --output /tmp/security_audit_docker.json

# 4. 네트워크 보안 스캔
echo "Running network security scan..."
nmap -sV -sC -O localhost > security_audit_network.txt

# 5. SSL/TLS 설정 검사
echo "Checking SSL/TLS configuration..."
testssl.sh --jsonfile security_audit_ssl.json https://patent2tech.com

echo "Security audit completed. Check the generated reports."
```

이러한 종합적인 배포 및 서비스화 계획을 통해 Patent2Tech LLM 시스템을 안정적이고 확장 가능한 프로덕션 서비스로 운영할 수 있습니다.

