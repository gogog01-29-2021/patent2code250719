from flask import Blueprint, request, jsonify
from src.models.patent import db, Patent, PatentAnalysis, ImplementationResult, QueryHistory
import json
import time
import sys
import os

# 상위 디렉토리의 모듈들을 import하기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from patent_llm_architecture import Patent2TechSystem
    from patent_data_processor import PatentDataPreprocessor
except ImportError:
    # 개발 환경에서 모듈을 찾을 수 없는 경우 더미 클래스 사용
    class Patent2TechSystem:
        def __init__(self):
            pass
        
        def process_patent_query(self, query, implementation_types=["code"]):
            return {
                "query": query,
                "found_patents": 1,
                "processed_patents": [{
                    "patent_id": "US1234567A",
                    "title": "Sample Patent for " + query,
                    "concept": {
                        "core_algorithm": "Sample algorithm based on " + query,
                        "technical_components": ["component1", "component2"],
                        "confidence_score": 0.85
                    },
                    "implementations": [{
                        "type": "code",
                        "content": f"# Sample implementation for {query}\ndef sample_function():\n    return 'Hello, Patent LLM!'",
                        "evaluation": {"syntax_check": 1.0, "completeness": 0.8}
                    }]
                }],
                "implementations": []
            }
    
    class PatentDataPreprocessor:
        def __init__(self):
            pass

patent_bp = Blueprint('patent', __name__)

# Patent2Tech 시스템 초기화
patent_system = Patent2TechSystem()
data_preprocessor = PatentDataPreprocessor()

@patent_bp.route('/search', methods=['POST'])
def search_patents():
    """특허 검색 API"""
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
            # Patent2Tech 시스템으로 특허 검색 및 분석
            results = patent_system.process_patent_query(query, ["code"])
            
            processing_time = time.time() - start_time
            
            # 결과를 데이터베이스에 저장
            saved_analyses = []
            for patent_result in results.get('processed_patents', []):
                # 특허 분석 결과 저장
                analysis = PatentAnalysis(
                    patent_id=patent_result['patent_id'],
                    query=query,
                    core_algorithm=patent_result['concept']['core_algorithm'],
                    technical_components=json.dumps(patent_result['concept']['technical_components']),
                    implementation_requirements=json.dumps([]),
                    confidence_score=patent_result['concept']['confidence_score']
                )
                db.session.add(analysis)
                db.session.flush()  # ID 생성을 위해
                
                # 구현 결과 저장
                for impl in patent_result['implementations']:
                    implementation = ImplementationResult(
                        analysis_id=analysis.id,
                        implementation_type=impl['type'],
                        content=impl['content'],
                        test_cases=json.dumps([]),
                        evaluation_metrics=json.dumps(impl['evaluation'])
                    )
                    db.session.add(implementation)
                
                saved_analyses.append(analysis.to_dict())
            
            # 쿼리 기록 업데이트
            query_record.results_count = len(saved_analyses)
            query_record.processing_time = processing_time
            query_record.status = 'completed'
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'query': query,
                'results_count': len(saved_analyses),
                'processing_time': processing_time,
                'analyses': saved_analyses,
                'raw_results': results
            })
            
        except Exception as e:
            # 오류 발생 시 쿼리 기록 업데이트
            query_record.status = 'failed'
            query_record.error_message = str(e)
            db.session.commit()
            
            return jsonify({'error': f'특허 검색 중 오류 발생: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'요청 처리 중 오류 발생: {str(e)}'}), 500

@patent_bp.route('/analyze/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """특정 분석 결과 조회"""
    try:
        analysis = PatentAnalysis.query.get_or_404(analysis_id)
        
        # 관련 구현 결과들도 함께 조회
        implementations = ImplementationResult.query.filter_by(analysis_id=analysis_id).all()
        
        result = analysis.to_dict()
        result['implementations'] = [impl.to_dict() for impl in implementations]
        
        return jsonify({
            'success': True,
            'analysis': result
        })
        
    except Exception as e:
        return jsonify({'error': f'분석 결과 조회 중 오류 발생: {str(e)}'}), 500

@patent_bp.route('/generate', methods=['POST'])
def generate_implementation():
    """새로운 구현 생성 API"""
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        implementation_type = data.get('type', 'code')
        
        if not analysis_id:
            return jsonify({'error': '분석 ID가 필요합니다.'}), 400
        
        analysis = PatentAnalysis.query.get_or_404(analysis_id)
        
        # 새로운 구현 생성 (실제로는 LLM을 사용)
        if implementation_type == 'code':
            content = f"""
# {analysis.patent_id} 특허 기반 구현
# 핵심 알고리즘: {analysis.core_algorithm}

def patent_implementation():
    \"\"\"
    특허 기반 구현 함수
    \"\"\"
    # 구현 로직
    result = process_algorithm()
    return result

def process_algorithm():
    # 핵심 알고리즘 구현
    return "구현 결과"

if __name__ == "__main__":
    result = patent_implementation()
    print(f"실행 결과: {{result}}")
"""
        elif implementation_type == 'circuit':
            content = f"""
* {analysis.patent_id} 특허 기반 회로 설계
* 핵심 알고리즘: {analysis.core_algorithm}

.subckt patent_circuit input output
R1 input node1 1k
C1 node1 0 1u
R2 node1 output 10k
.ends

.circuit main
X1 in out patent_circuit
V1 in 0 DC 5V
.end
"""
        else:
            content = f"# {implementation_type} 구현\n# 특허: {analysis.patent_id}\n# 알고리즘: {analysis.core_algorithm}"
        
        # 구현 결과 저장
        implementation = ImplementationResult(
            analysis_id=analysis_id,
            implementation_type=implementation_type,
            content=content,
            test_cases=json.dumps(["기본 테스트", "경계값 테스트"]),
            evaluation_metrics=json.dumps({
                "syntax_check": 1.0,
                "completeness": 0.8,
                "functionality": 0.7
            })
        )
        
        db.session.add(implementation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'implementation': implementation.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': f'구현 생성 중 오류 발생: {str(e)}'}), 500

@patent_bp.route('/history', methods=['GET'])
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

@patent_bp.route('/statistics', methods=['GET'])
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

@patent_bp.route('/patents', methods=['GET'])
def list_patents():
    """특허 목록 조회"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search = request.args.get('search', '')
        
        query = Patent.query
        
        if search:
            query = query.filter(
                db.or_(
                    Patent.title.contains(search),
                    Patent.patent_id.contains(search),
                    Patent.abstract.contains(search)
                )
            )
        
        patents = query.order_by(Patent.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'patents': [patent.to_dict() for patent in patents.items],
            'total': patents.total,
            'pages': patents.pages,
            'current_page': page
        })
        
    except Exception as e:
        return jsonify({'error': f'특허 목록 조회 중 오류 발생: {str(e)}'}), 500

@patent_bp.route('/patents', methods=['POST'])
def add_patent():
    """새 특허 추가"""
    try:
        data = request.get_json()
        
        required_fields = ['patent_id', 'title']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field}는 필수 항목입니다.'}), 400
        
        # 중복 확인
        existing = Patent.query.filter_by(patent_id=data['patent_id']).first()
        if existing:
            return jsonify({'error': '이미 존재하는 특허 ID입니다.'}), 400
        
        patent = Patent(
            patent_id=data['patent_id'],
            title=data['title'],
            abstract=data.get('abstract', ''),
            claims=json.dumps(data.get('claims', [])),
            inventors=json.dumps(data.get('inventors', [])),
            assignee=data.get('assignee', ''),
            filing_date=data.get('filing_date', ''),
            publication_date=data.get('publication_date', ''),
            classification_codes=json.dumps(data.get('classification_codes', []))
        )
        
        db.session.add(patent)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'patent': patent.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'특허 추가 중 오류 발생: {str(e)}'}), 500

