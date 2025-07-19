from flask import Blueprint, request, jsonify
from src.models.patent import db, Patent, PatentAnalysis, ImplementationResult, QueryHistory
import json
import time

test_patent_bp = Blueprint('test_patent', __name__)

@test_patent_bp.route('/search', methods=['POST'])
def search_patents():
    """특허 검색 API - 테스트 버전"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': '검색 쿼리가 필요합니다.'}), 400
        
        # 쿼리 기록 저장
        query_record = QueryHistory(query_text=query, status='pending')
        db.session.add(query_record)
        db.session.commit()
        
        start_time = time.time()
        
        try:
            # 샘플 결과 생성 (실제 LLM 대신)
            sample_patent = {
                "patent_id": "US1234567A",
                "title": f"Sample Patent for {query}",
                "concept": {
                    "core_algorithm": f"Sample algorithm based on {query}",
                    "technical_components": ["component1", "component2", "AI module"],
                    "confidence_score": 0.85
                },
                "implementations": [{
                    "type": "code",
                    "content": f"""# {query} 구현
import numpy as np
import tensorflow as tf

class PatentImplementation:
    def __init__(self):
        self.model = None
        
    def build_model(self):
        # {query}를 위한 모델 구축
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model = model
        return model
    
    def predict(self, data):
        if self.model is None:
            self.build_model()
        return self.model.predict(data)

# 사용 예시
if __name__ == "__main__":
    impl = PatentImplementation()
    model = impl.build_model()
    print("모델이 성공적으로 구축되었습니다.")
""",
                    "evaluation": {"syntax_check": 1.0, "completeness": 0.8, "functionality": 0.7}
                }]
            }
            
            processing_time = time.time() - start_time
            
            # 특허 분석 결과 저장
            analysis = PatentAnalysis(
                patent_id=sample_patent['patent_id'],
                query=query,
                core_algorithm=sample_patent['concept']['core_algorithm'],
                technical_components=json.dumps(sample_patent['concept']['technical_components']),
                implementation_requirements=json.dumps([]),
                confidence_score=sample_patent['concept']['confidence_score']
            )
            db.session.add(analysis)
            db.session.flush()  # ID 생성을 위해
            
            # 구현 결과 저장
            for impl in sample_patent['implementations']:
                implementation = ImplementationResult(
                    analysis_id=analysis.id,
                    implementation_type=impl['type'],
                    content=impl['content'],
                    test_cases=json.dumps(["기본 테스트", "성능 테스트"]),
                    evaluation_metrics=json.dumps(impl['evaluation'])
                )
                db.session.add(implementation)
            
            # 쿼리 기록 업데이트
            query_record.results_count = 1
            query_record.processing_time = processing_time
            query_record.status = 'completed'
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'query': query,
                'found_patents': 1,
                'results_count': 1,
                'processing_time': processing_time,
                'analyses': [analysis.to_dict()],
                'raw_results': {
                    'processed_patents': [sample_patent],
                    'implementations': []
                }
            })
            
        except Exception as e:
            # 오류 발생 시 쿼리 기록 업데이트
            query_record.status = 'failed'
            query_record.error_message = str(e)
            db.session.commit()
            
            return jsonify({'error': f'특허 검색 중 오류 발생: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'요청 처리 중 오류 발생: {str(e)}'}), 500

@test_patent_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """시스템 통계 조회"""
    try:
        total_queries = QueryHistory.query.count()
        successful_queries = QueryHistory.query.filter_by(status='completed').count()
        total_analyses = PatentAnalysis.query.count()
        total_implementations = ImplementationResult.query.count()
        
        # 구현 타입별 통계
        impl_types = db.session.query(
            ImplementationResult.implementation_type,
            db.func.count(ImplementationResult.id)
        ).group_by(ImplementationResult.implementation_type).all()
        
        # 최근 평균 처리 시간
        recent_queries = QueryHistory.query.filter(
            QueryHistory.status == 'completed',
            QueryHistory.processing_time.isnot(None)
        ).order_by(QueryHistory.created_at.desc()).limit(10).all()
        
        avg_processing_time = sum(q.processing_time for q in recent_queries) / len(recent_queries) if recent_queries else 0
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0,
                'total_analyses': total_analyses,
                'total_implementations': total_implementations,
                'implementation_types': dict(impl_types),
                'average_processing_time': round(avg_processing_time, 2)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'통계 조회 중 오류 발생: {str(e)}'}), 500

@test_patent_bp.route('/history', methods=['GET'])
def get_query_history():
    """쿼리 기록 조회"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        queries = QueryHistory.query.order_by(QueryHistory.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'queries': [query.to_dict() for query in queries.items],
            'total': queries.total,
            'pages': queries.pages,
            'current_page': page
        })
        
    except Exception as e:
        return jsonify({'error': f'기록 조회 중 오류 발생: {str(e)}'}), 500

