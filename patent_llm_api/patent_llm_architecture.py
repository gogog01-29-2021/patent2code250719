#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patent2Tech LLM 시스템 아키텍처
Patent2Tech LLM System Architecture

특허 문서에서 기술 개념을 추출하고 구현 가능한 형태로 변환하는 LLM 시스템

MIT License
Copyright (c) 2025
"""

import os
import json
import sqlite3
import requests
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import re
import base64
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PatentDocument:
    """특허 문서 데이터 클래스"""
    patent_id: str
    title: str
    abstract: str
    claims: List[str]
    description: str
    figures: List[str]  # 도면 파일 경로들
    inventors: List[str]
    assignee: str
    filing_date: str
    publication_date: str
    classification_codes: List[str]

@dataclass
class TechnicalConcept:
    """기술 개념 데이터 클래스"""
    concept_id: str
    patent_id: str
    core_algorithm: str
    technical_components: List[str]
    implementation_requirements: List[str]
    related_figures: List[str]
    confidence_score: float

@dataclass
class ImplementationResult:
    """구현 결과 데이터 클래스"""
    result_id: str
    concept_id: str
    implementation_type: str  # "code", "circuit", "cad", "simulation"
    content: str
    test_cases: List[str]
    evaluation_metrics: Dict[str, float]
    generated_at: str

class PatentRetriever(ABC):
    """특허 검색기 추상 클래스"""
    
    @abstractmethod
    def search_similar_patents(self, query: str, top_k: int = 10) -> List[PatentDocument]:
        """유사 특허 검색"""
        pass
    
    @abstractmethod
    def get_patent_by_id(self, patent_id: str) -> Optional[PatentDocument]:
        """특허 ID로 특허 문서 조회"""
        pass

class BM25Retriever(PatentRetriever):
    """BM25 기반 특허 검색기"""
    
    def __init__(self, patent_db_path: str):
        self.db_path = patent_db_path
        self.init_database()
    
    def init_database(self):
        """특허 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patents (
                patent_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                claims TEXT,
                description TEXT,
                figures TEXT,
                inventors TEXT,
                assignee TEXT,
                filing_date TEXT,
                publication_date TEXT,
                classification_codes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patent_embeddings (
                patent_id TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patent_id) REFERENCES patents (patent_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("특허 데이터베이스 초기화 완료")
    
    def search_similar_patents(self, query: str, top_k: int = 10) -> List[PatentDocument]:
        """BM25 기반 유사 특허 검색"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 간단한 키워드 기반 검색 (실제로는 BM25 알고리즘 구현 필요)
        search_terms = query.lower().split()
        search_condition = " OR ".join([f"(title LIKE '%{term}%' OR abstract LIKE '%{term}%' OR claims LIKE '%{term}%')" for term in search_terms])
        
        cursor.execute(f'''
            SELECT * FROM patents 
            WHERE {search_condition}
            LIMIT ?
        ''', (top_k,))
        
        results = cursor.fetchall()
        conn.close()
        
        patents = []
        for row in results:
            patent = PatentDocument(
                patent_id=row[0],
                title=row[1],
                abstract=row[2],
                claims=json.loads(row[3]) if row[3] else [],
                description=row[4],
                figures=json.loads(row[5]) if row[5] else [],
                inventors=json.loads(row[6]) if row[6] else [],
                assignee=row[7],
                filing_date=row[8],
                publication_date=row[9],
                classification_codes=json.loads(row[10]) if row[10] else []
            )
            patents.append(patent)
        
        return patents
    
    def get_patent_by_id(self, patent_id: str) -> Optional[PatentDocument]:
        """특허 ID로 특허 문서 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patents WHERE patent_id = ?', (patent_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PatentDocument(
                patent_id=row[0],
                title=row[1],
                abstract=row[2],
                claims=json.loads(row[3]) if row[3] else [],
                description=row[4],
                figures=json.loads(row[5]) if row[5] else [],
                inventors=json.loads(row[6]) if row[6] else [],
                assignee=row[7],
                filing_date=row[8],
                publication_date=row[9],
                classification_codes=json.loads(row[10]) if row[10] else []
            )
        return None

class PatentReconstructor:
    """특허 기술 개념 재구축기"""
    
    def __init__(self, llm_provider: str = "openai", model_name: str = "gpt-4"):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
    
    def analyze_patent_claims(self, patent: PatentDocument) -> TechnicalConcept:
        """특허 클레임 분석 및 기술 개념 추출"""
        
        prompt = f"""
다음 특허 문서를 분석하여 핵심 기술 개념을 추출해주세요:

특허 제목: {patent.title}
특허 초록: {patent.abstract}
주요 클레임: {' '.join(patent.claims[:3])}

분석 항목:
1. 핵심 알고리즘 또는 기술적 원리
2. 주요 기술 구성 요소
3. 구현을 위한 요구사항
4. 관련 도면 설명

JSON 형식으로 응답해주세요:
{{
  "core_algorithm": "핵심 알고리즘 설명",
  "technical_components": ["구성요소1", "구성요소2", "구성요소3"],
  "implementation_requirements": ["요구사항1", "요구사항2", "요구사항3"],
  "confidence_score": 0.85
}}
"""
        
        try:
            if self.llm_provider == "openai":
                response = self._call_openai_api(prompt)
            else:
                response = self._call_local_llm(prompt)
            
            result = json.loads(response)
            
            concept = TechnicalConcept(
                concept_id=f"concept_{patent.patent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                patent_id=patent.patent_id,
                core_algorithm=result.get("core_algorithm", ""),
                technical_components=result.get("technical_components", []),
                implementation_requirements=result.get("implementation_requirements", []),
                related_figures=patent.figures,
                confidence_score=result.get("confidence_score", 0.5)
            )
            
            return concept
            
        except Exception as e:
            logger.error(f"특허 클레임 분석 오류: {e}")
            return None
    
    def _call_openai_api(self, prompt: str) -> str:
        """OpenAI API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API 오류: {response.status_code}")
    
    def _call_local_llm(self, prompt: str) -> str:
        """로컬 LLM 호출 (Ollama 등)"""
        # 로컬 LLM 호출 로직 구현
        # 여기서는 간단한 예시만 제공
        return '{"core_algorithm": "분석 중", "technical_components": [], "implementation_requirements": [], "confidence_score": 0.5}'

class PatentGenerator:
    """특허 기반 구현 생성기"""
    
    def __init__(self, llm_provider: str = "openai", model_name: str = "gpt-4"):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
    
    def generate_python_implementation(self, concept: TechnicalConcept) -> ImplementationResult:
        """Python 구현 코드 생성"""
        
        prompt = f"""
다음 기술 개념을 바탕으로 Python 구현 코드를 생성해주세요:

핵심 알고리즘: {concept.core_algorithm}
기술 구성 요소: {', '.join(concept.technical_components)}
구현 요구사항: {', '.join(concept.implementation_requirements)}

요구사항:
1. 완전히 실행 가능한 Python 코드
2. 적절한 주석과 문서화
3. 테스트 케이스 포함
4. 필요한 라이브러리 import

코드만 응답해주세요:
"""
        
        try:
            if self.llm_provider == "openai":
                code = self._call_openai_api(prompt)
            else:
                code = self._call_local_llm(prompt)
            
            # 테스트 케이스 생성
            test_cases = self._generate_test_cases(concept, code)
            
            result = ImplementationResult(
                result_id=f"impl_{concept.concept_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                concept_id=concept.concept_id,
                implementation_type="code",
                content=code,
                test_cases=test_cases,
                evaluation_metrics={"syntax_check": 1.0, "completeness": 0.8},
                generated_at=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Python 구현 생성 오류: {e}")
            return None
    
    def generate_circuit_design(self, concept: TechnicalConcept) -> ImplementationResult:
        """회로 설계 생성"""
        
        prompt = f"""
다음 기술 개념을 바탕으로 회로 설계를 생성해주세요:

핵심 알고리즘: {concept.core_algorithm}
기술 구성 요소: {', '.join(concept.technical_components)}

SPICE 넷리스트 또는 회로 설명을 제공해주세요:
"""
        
        try:
            if self.llm_provider == "openai":
                circuit = self._call_openai_api(prompt)
            else:
                circuit = self._call_local_llm(prompt)
            
            result = ImplementationResult(
                result_id=f"circuit_{concept.concept_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                concept_id=concept.concept_id,
                implementation_type="circuit",
                content=circuit,
                test_cases=[],
                evaluation_metrics={"feasibility": 0.7},
                generated_at=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"회로 설계 생성 오류: {e}")
            return None
    
    def _generate_test_cases(self, concept: TechnicalConcept, code: str) -> List[str]:
        """테스트 케이스 생성"""
        test_prompt = f"""
다음 코드에 대한 테스트 케이스를 생성해주세요:

{code[:1000]}...

간단한 테스트 케이스 3개를 생성해주세요.
"""
        
        try:
            if self.llm_provider == "openai":
                tests = self._call_openai_api(test_prompt)
                return [tests]
            else:
                return ["# 테스트 케이스 생성 중"]
        except:
            return ["# 테스트 케이스 생성 실패"]
    
    def _call_openai_api(self, prompt: str) -> str:
        """OpenAI API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API 오류: {response.status_code}")
    
    def _call_local_llm(self, prompt: str) -> str:
        """로컬 LLM 호출"""
        return "# 로컬 LLM 구현 필요"

class PatentEvaluator:
    """특허 구현 평가기"""
    
    def __init__(self):
        self.evaluation_db_path = "patent_evaluations.db"
        self.init_database()
    
    def init_database(self):
        """평가 데이터베이스 초기화"""
        conn = sqlite3.connect(self.evaluation_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                result_id TEXT,
                evaluation_type TEXT,
                metrics TEXT,
                score REAL,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def evaluate_implementation(self, result: ImplementationResult) -> Dict[str, float]:
        """구현 결과 평가"""
        
        metrics = {}
        
        if result.implementation_type == "code":
            metrics = self._evaluate_code(result.content)
        elif result.implementation_type == "circuit":
            metrics = self._evaluate_circuit(result.content)
        
        # 평가 결과 저장
        self._save_evaluation(result.result_id, "implementation", metrics)
        
        return metrics
    
    def _evaluate_code(self, code: str) -> Dict[str, float]:
        """코드 평가"""
        metrics = {
            "syntax_check": 0.0,
            "completeness": 0.0,
            "functionality": 0.0,
            "documentation": 0.0
        }
        
        # 구문 검사
        try:
            compile(code, '<string>', 'exec')
            metrics["syntax_check"] = 1.0
        except SyntaxError:
            metrics["syntax_check"] = 0.0
        
        # 완성도 검사 (간단한 휴리스틱)
        if "def " in code and "import " in code:
            metrics["completeness"] = 0.8
        elif "def " in code:
            metrics["completeness"] = 0.6
        else:
            metrics["completeness"] = 0.3
        
        # 문서화 검사
        if '"""' in code or "# " in code:
            metrics["documentation"] = 0.7
        else:
            metrics["documentation"] = 0.2
        
        # 기능성은 실제 실행 테스트가 필요하므로 기본값
        metrics["functionality"] = 0.6
        
        return metrics
    
    def _evaluate_circuit(self, circuit: str) -> Dict[str, float]:
        """회로 평가"""
        metrics = {
            "feasibility": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0
        }
        
        # 간단한 회로 평가 로직
        if any(keyword in circuit.lower() for keyword in ["resistor", "capacitor", "transistor", "voltage", "current"]):
            metrics["feasibility"] = 0.7
            metrics["completeness"] = 0.6
            metrics["accuracy"] = 0.5
        
        return metrics
    
    def _save_evaluation(self, result_id: str, eval_type: str, metrics: Dict[str, float]):
        """평가 결과 저장"""
        conn = sqlite3.connect(self.evaluation_db_path)
        cursor = conn.cursor()
        
        evaluation_id = f"eval_{result_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        avg_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        cursor.execute('''
            INSERT INTO evaluations (evaluation_id, result_id, evaluation_type, metrics, score)
            VALUES (?, ?, ?, ?, ?)
        ''', (evaluation_id, result_id, eval_type, json.dumps(metrics), avg_score))
        
        conn.commit()
        conn.close()

class Patent2TechSystem:
    """Patent2Tech 통합 시스템"""
    
    def __init__(self, patent_db_path: str = "patents.db"):
        self.retriever = BM25Retriever(patent_db_path)
        self.reconstructor = PatentReconstructor()
        self.generator = PatentGenerator()
        self.evaluator = PatentEvaluator()
        
        self.results_db_path = "patent2tech_results.db"
        self.init_results_database()
    
    def init_results_database(self):
        """결과 데이터베이스 초기화"""
        conn = sqlite3.connect(self.results_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_concepts (
                concept_id TEXT PRIMARY KEY,
                patent_id TEXT,
                core_algorithm TEXT,
                technical_components TEXT,
                implementation_requirements TEXT,
                related_figures TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS implementation_results (
                result_id TEXT PRIMARY KEY,
                concept_id TEXT,
                implementation_type TEXT,
                content TEXT,
                test_cases TEXT,
                evaluation_metrics TEXT,
                generated_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_patent_query(self, query: str, implementation_types: List[str] = ["code"]) -> Dict:
        """특허 쿼리 처리 및 구현 생성"""
        
        logger.info(f"특허 쿼리 처리 시작: {query}")
        
        # 1. 특허 검색
        patents = self.retriever.search_similar_patents(query, top_k=5)
        if not patents:
            return {"error": "관련 특허를 찾을 수 없습니다."}
        
        results = {
            "query": query,
            "found_patents": len(patents),
            "processed_patents": [],
            "implementations": []
        }
        
        # 2. 각 특허에 대해 처리
        for patent in patents[:3]:  # 상위 3개 특허만 처리
            logger.info(f"특허 처리 중: {patent.patent_id}")
            
            # 기술 개념 재구축
            concept = self.reconstructor.analyze_patent_claims(patent)
            if not concept:
                continue
            
            # 기술 개념 저장
            self._save_technical_concept(concept)
            
            patent_result = {
                "patent_id": patent.patent_id,
                "title": patent.title,
                "concept": {
                    "core_algorithm": concept.core_algorithm,
                    "technical_components": concept.technical_components,
                    "confidence_score": concept.confidence_score
                },
                "implementations": []
            }
            
            # 3. 구현 생성
            for impl_type in implementation_types:
                if impl_type == "code":
                    impl_result = self.generator.generate_python_implementation(concept)
                elif impl_type == "circuit":
                    impl_result = self.generator.generate_circuit_design(concept)
                else:
                    continue
                
                if impl_result:
                    # 구현 평가
                    evaluation = self.evaluator.evaluate_implementation(impl_result)
                    impl_result.evaluation_metrics.update(evaluation)
                    
                    # 구현 결과 저장
                    self._save_implementation_result(impl_result)
                    
                    patent_result["implementations"].append({
                        "type": impl_result.implementation_type,
                        "content": impl_result.content[:500] + "..." if len(impl_result.content) > 500 else impl_result.content,
                        "evaluation": impl_result.evaluation_metrics
                    })
                    
                    results["implementations"].append(impl_result)
            
            results["processed_patents"].append(patent_result)
        
        logger.info(f"특허 쿼리 처리 완료: {len(results['implementations'])}개 구현 생성")
        return results
    
    def _save_technical_concept(self, concept: TechnicalConcept):
        """기술 개념 저장"""
        conn = sqlite3.connect(self.results_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO technical_concepts 
            (concept_id, patent_id, core_algorithm, technical_components, 
             implementation_requirements, related_figures, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            concept.concept_id,
            concept.patent_id,
            concept.core_algorithm,
            json.dumps(concept.technical_components),
            json.dumps(concept.implementation_requirements),
            json.dumps(concept.related_figures),
            concept.confidence_score
        ))
        
        conn.commit()
        conn.close()
    
    def _save_implementation_result(self, result: ImplementationResult):
        """구현 결과 저장"""
        conn = sqlite3.connect(self.results_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO implementation_results 
            (result_id, concept_id, implementation_type, content, 
             test_cases, evaluation_metrics, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.result_id,
            result.concept_id,
            result.implementation_type,
            result.content,
            json.dumps(result.test_cases),
            json.dumps(result.evaluation_metrics),
            result.generated_at
        ))
        
        conn.commit()
        conn.close()
    
    def get_system_statistics(self) -> Dict:
        """시스템 통계 조회"""
        conn = sqlite3.connect(self.results_db_path)
        cursor = conn.cursor()
        
        # 기술 개념 통계
        cursor.execute("SELECT COUNT(*) FROM technical_concepts")
        total_concepts = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence_score) FROM technical_concepts")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # 구현 결과 통계
        cursor.execute("SELECT implementation_type, COUNT(*) FROM implementation_results GROUP BY implementation_type")
        impl_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_concepts": total_concepts,
            "average_confidence": round(avg_confidence, 2),
            "implementation_statistics": dict(impl_stats)
        }

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🔬 Patent2Tech LLM 시스템")
    print("특허에서 기술 구현으로 - From Patent to Implementation")
    print("=" * 80)
    
    # 시스템 초기화
    system = Patent2TechSystem()
    
    while True:
        print("\n📋 메뉴를 선택하세요:")
        print("1. 🔍 특허 검색 및 구현 생성")
        print("2. 📊 시스템 통계 조회")
        print("3. ❌ 종료")
        
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == '1':
            query = input("\n검색할 기술 또는 특허 내용을 입력하세요: ").strip()
            if query:
                print(f"\n🔍 '{query}' 검색 중...")
                
                impl_types = ["code"]
                circuit_choice = input("회로 설계도 생성하시겠습니까? (y/n): ").strip().lower()
                if circuit_choice == 'y':
                    impl_types.append("circuit")
                
                results = system.process_patent_query(query, impl_types)
                
                if "error" in results:
                    print(f"❌ {results['error']}")
                else:
                    print(f"\n✅ {results['found_patents']}개 특허 발견, {len(results['implementations'])}개 구현 생성")
                    
                    for i, patent in enumerate(results['processed_patents'], 1):
                        print(f"\n📄 특허 {i}: {patent['title']}")
                        print(f"   핵심 알고리즘: {patent['concept']['core_algorithm'][:100]}...")
                        print(f"   신뢰도: {patent['concept']['confidence_score']:.2f}")
                        
                        for impl in patent['implementations']:
                            print(f"   🔧 {impl['type']} 구현 생성됨 (평가점수: {sum(impl['evaluation'].values())/len(impl['evaluation']):.2f})")
        
        elif choice == '2':
            stats = system.get_system_statistics()
            print(f"\n📊 시스템 통계:")
            print(f"- 총 기술 개념: {stats['total_concepts']}개")
            print(f"- 평균 신뢰도: {stats['average_confidence']}")
            print(f"- 구현 통계: {stats['implementation_statistics']}")
        
        elif choice == '3':
            print("시스템을 종료합니다.")
            break
        
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()

