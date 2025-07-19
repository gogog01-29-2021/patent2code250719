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
