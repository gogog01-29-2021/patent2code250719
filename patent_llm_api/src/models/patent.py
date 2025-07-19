from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Patent(db.Model):
    """특허 모델"""
    __tablename__ = 'patents'
    
    id = db.Column(db.Integer, primary_key=True)
    patent_id = db.Column(db.String(50), unique=True, nullable=False)
    title = db.Column(db.Text, nullable=False)
    abstract = db.Column(db.Text)
    claims = db.Column(db.Text)  # JSON 형태로 저장
    inventors = db.Column(db.Text)  # JSON 형태로 저장
    assignee = db.Column(db.String(200))
    filing_date = db.Column(db.String(20))
    publication_date = db.Column(db.String(20))
    classification_codes = db.Column(db.Text)  # JSON 형태로 저장
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patent_id': self.patent_id,
            'title': self.title,
            'abstract': self.abstract,
            'claims': json.loads(self.claims) if self.claims else [],
            'inventors': json.loads(self.inventors) if self.inventors else [],
            'assignee': self.assignee,
            'filing_date': self.filing_date,
            'publication_date': self.publication_date,
            'classification_codes': json.loads(self.classification_codes) if self.classification_codes else [],
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class PatentAnalysis(db.Model):
    """특허 분석 결과 모델"""
    __tablename__ = 'patent_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    patent_id = db.Column(db.String(50), nullable=False)
    query = db.Column(db.Text, nullable=False)
    core_algorithm = db.Column(db.Text)
    technical_components = db.Column(db.Text)  # JSON 형태
    implementation_requirements = db.Column(db.Text)  # JSON 형태
    confidence_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patent_id': self.patent_id,
            'query': self.query,
            'core_algorithm': self.core_algorithm,
            'technical_components': json.loads(self.technical_components) if self.technical_components else [],
            'implementation_requirements': json.loads(self.implementation_requirements) if self.implementation_requirements else [],
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ImplementationResult(db.Model):
    """구현 결과 모델"""
    __tablename__ = 'implementation_results'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('patent_analyses.id'), nullable=False)
    implementation_type = db.Column(db.String(50), nullable=False)  # 'code', 'circuit', 'cad'
    content = db.Column(db.Text, nullable=False)
    test_cases = db.Column(db.Text)  # JSON 형태
    evaluation_metrics = db.Column(db.Text)  # JSON 형태
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    analysis = db.relationship('PatentAnalysis', backref=db.backref('implementations', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'implementation_type': self.implementation_type,
            'content': self.content,
            'test_cases': json.loads(self.test_cases) if self.test_cases else [],
            'evaluation_metrics': json.loads(self.evaluation_metrics) if self.evaluation_metrics else {},
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class QueryHistory(db.Model):
    """쿼리 기록 모델"""
    __tablename__ = 'query_history'
    
    id = db.Column(db.Integer, primary_key=True)
    query_text = db.Column(db.Text, nullable=False)
    results_count = db.Column(db.Integer, default=0)
    processing_time = db.Column(db.Float)  # 초 단위
    status = db.Column(db.String(20), default='pending')  # 'pending', 'completed', 'failed'
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'query_text': self.query_text,
            'results_count': self.results_count,
            'processing_time': self.processing_time,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

