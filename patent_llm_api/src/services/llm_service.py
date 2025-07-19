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