#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특허 3D 이미지 처리 모듈
Patent 3D Image Processing Module

특허 도면에서 3D 정보를 추출하고 3D 모델을 재구성하는 모듈

MIT License
Copyright (c) 2025
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import base64
from datetime import datetime
import sqlite3

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DiagramElement:
    """도면 요소 데이터 클래스"""
    element_id: str
    element_type: str  # 'line', 'circle', 'rectangle', 'text', 'arrow'
    coordinates: List[Tuple[int, int]]
    properties: Dict[str, Any]
    confidence: float

@dataclass
class Reconstructed3DModel:
    """재구성된 3D 모델 데이터 클래스"""
    model_id: str
    patent_id: str
    source_figures: List[str]
    model_description: str
    vertices: List[Tuple[float, float, float]]
    faces: List[List[int]]
    materials: Dict[str, Any]
    reconstruction_method: str
    confidence_score: float

class DiagramAnalyzer:
    """도면 분석기"""
    
    def __init__(self):
        self.element_detectors = {
            'line': self._detect_lines,
            'circle': self._detect_circles,
            'rectangle': self._detect_rectangles,
            'text': self._detect_text_regions,
            'arrow': self._detect_arrows
        }
    
    def analyze_diagram(self, image_path: str) -> List[DiagramElement]:
        """도면 분석 및 요소 추출"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"이미지 로드 실패: {image_path}")
                return []
            
            # 전처리
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            elements = []
            
            # 각 요소 타입별 검출
            for element_type, detector in self.element_detectors.items():
                detected_elements = detector(gray, image)
                elements.extend(detected_elements)
            
            logger.info(f"도면 분석 완료: {len(elements)}개 요소 검출")
            return elements
            
        except Exception as e:
            logger.error(f"도면 분석 오류: {e}")
            return []
    
    def _detect_lines(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[DiagramElement]:
        """직선 검출"""
        elements = []
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Hough 직선 변환
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                
                # 직선 속성 계산
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                element = DiagramElement(
                    element_id=f"line_{i}",
                    element_type="line",
                    coordinates=[(x1, y1), (x2, y2)],
                    properties={
                        "length": length,
                        "angle": angle,
                        "thickness": 1  # 기본값
                    },
                    confidence=0.8
                )
                elements.append(element)
        
        return elements
    
    def _detect_circles(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[DiagramElement]:
        """원 검출"""
        elements = []
        
        # Hough 원 변환
        circles = cv2.HoughCircles(
            gray_image, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles):
                element = DiagramElement(
                    element_id=f"circle_{i}",
                    element_type="circle",
                    coordinates=[(x, y)],
                    properties={
                        "radius": r,
                        "area": np.pi * r * r,
                        "center": (x, y)
                    },
                    confidence=0.7
                )
                elements.append(element)
        
        return elements
    
    def _detect_rectangles(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[DiagramElement]:
        """사각형 검출"""
        elements = []
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # 윤곽선 근사
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 사각형 판별 (4개 꼭짓점)
            if len(approx) == 4:
                # 면적 필터링
                area = cv2.contourArea(contour)
                if area > 100:  # 최소 면적
                    
                    # 바운딩 박스
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    element = DiagramElement(
                        element_id=f"rectangle_{i}",
                        element_type="rectangle",
                        coordinates=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                        properties={
                            "width": w,
                            "height": h,
                            "area": area,
                            "aspect_ratio": w/h if h > 0 else 0
                        },
                        confidence=0.6
                    )
                    elements.append(element)
        
        return elements
    
    def _detect_text_regions(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[DiagramElement]:
        """텍스트 영역 검출"""
        elements = []
        
        try:
            # MSER (Maximally Stable Extremal Regions) 검출기 사용
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_image)
            
            for i, region in enumerate(regions):
                # 바운딩 박스 계산
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                
                # 텍스트 영역 필터링 (크기 및 종횡비 기준)
                if w > 10 and h > 5 and 0.1 < h/w < 10:
                    element = DiagramElement(
                        element_id=f"text_{i}",
                        element_type="text",
                        coordinates=[(x, y), (x+w, y+h)],
                        properties={
                            "width": w,
                            "height": h,
                            "aspect_ratio": w/h
                        },
                        confidence=0.5
                    )
                    elements.append(element)
        
        except Exception as e:
            logger.warning(f"텍스트 영역 검출 오류: {e}")
        
        return elements
    
    def _detect_arrows(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[DiagramElement]:
        """화살표 검출"""
        elements = []
        
        # 화살표 검출은 복잡하므로 간단한 휴리스틱 사용
        # 실제로는 더 정교한 알고리즘이 필요
        
        # 직선 검출 후 끝점에서 화살표 머리 검출
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                
                # 화살표 머리 검출 로직 (간단한 버전)
                # 실제로는 더 복잡한 패턴 매칭이 필요
                if self._is_arrow_like(gray_image, x1, y1, x2, y2):
                    element = DiagramElement(
                        element_id=f"arrow_{i}",
                        element_type="arrow",
                        coordinates=[(x1, y1), (x2, y2)],
                        properties={
                            "direction": np.arctan2(y2-y1, x2-x1) * 180 / np.pi,
                            "length": np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        },
                        confidence=0.4
                    )
                    elements.append(element)
        
        return elements
    
    def _is_arrow_like(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """화살표 유사성 판별 (간단한 휴리스틱)"""
        # 실제로는 더 정교한 패턴 매칭 알고리즘 필요
        return np.random.random() > 0.8  # 임시 구현

class Diagram2ModelConverter:
    """도면-3D모델 변환기"""
    
    def __init__(self):
        self.conversion_methods = {
            'orthographic_projection': self._orthographic_reconstruction,
            'isometric_reconstruction': self._isometric_reconstruction,
            'perspective_analysis': self._perspective_reconstruction,
            'multi_view_synthesis': self._multi_view_reconstruction
        }
    
    def convert_diagram_to_3d(self, diagram_elements: List[DiagramElement], 
                            conversion_method: str = 'orthographic_projection') -> Optional[Reconstructed3DModel]:
        """도면을 3D 모델로 변환"""
        
        if conversion_method not in self.conversion_methods:
            logger.error(f"지원하지 않는 변환 방법: {conversion_method}")
            return None
        
        try:
            converter = self.conversion_methods[conversion_method]
            model = converter(diagram_elements)
            
            if model:
                logger.info(f"3D 모델 변환 완료: {model.model_id}")
            
            return model
            
        except Exception as e:
            logger.error(f"3D 모델 변환 오류: {e}")
            return None
    
    def _orthographic_reconstruction(self, elements: List[DiagramElement]) -> Optional[Reconstructed3DModel]:
        """직교 투영 기반 3D 재구성"""
        
        # 간단한 예시: 사각형을 3D 박스로 변환
        vertices = []
        faces = []
        
        rectangles = [e for e in elements if e.element_type == 'rectangle']
        
        if rectangles:
            rect = rectangles[0]  # 첫 번째 사각형 사용
            coords = rect.coordinates
            
            # 2D 사각형을 3D 박스로 확장
            x1, y1 = coords[0]
            x2, y2 = coords[2]
            
            # Z축 깊이 추정 (간단한 휴리스틱)
            depth = abs(x2 - x1) * 0.5
            
            # 8개 꼭짓점 생성
            vertices = [
                (x1, y1, 0), (x2, y1, 0), (x2, y2, 0), (x1, y2, 0),  # 앞면
                (x1, y1, depth), (x2, y1, depth), (x2, y2, depth), (x1, y2, depth)  # 뒷면
            ]
            
            # 6개 면 정의
            faces = [
                [0, 1, 2, 3],  # 앞면
                [4, 7, 6, 5],  # 뒷면
                [0, 4, 5, 1],  # 아래면
                [2, 6, 7, 3],  # 위면
                [0, 3, 7, 4],  # 왼쪽면
                [1, 5, 6, 2]   # 오른쪽면
            ]
            
            model = Reconstructed3DModel(
                model_id=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                patent_id="",
                source_figures=[],
                model_description="Orthographic projection reconstruction of rectangular element",
                vertices=vertices,
                faces=faces,
                materials={"default": {"color": [0.7, 0.7, 0.7]}},
                reconstruction_method="orthographic_projection",
                confidence_score=0.6
            )
            
            return model
        
        return None
    
    def _isometric_reconstruction(self, elements: List[DiagramElement]) -> Optional[Reconstructed3DModel]:
        """등각 투영 기반 3D 재구성"""
        # 등각 투영 도면에서 3D 정보 추출
        # 실제 구현에서는 더 복잡한 기하학적 계산 필요
        
        lines = [e for e in elements if e.element_type == 'line']
        
        if len(lines) >= 3:
            # 간단한 3D 구조 생성
            vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
            
            faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                    [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
            
            model = Reconstructed3DModel(
                model_id=f"iso_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                patent_id="",
                source_figures=[],
                model_description="Isometric reconstruction",
                vertices=vertices,
                faces=faces,
                materials={"default": {"color": [0.5, 0.8, 0.5]}},
                reconstruction_method="isometric_reconstruction",
                confidence_score=0.5
            )
            
            return model
        
        return None
    
    def _perspective_reconstruction(self, elements: List[DiagramElement]) -> Optional[Reconstructed3DModel]:
        """원근법 기반 3D 재구성"""
        # 원근법 도면에서 소실점을 찾고 3D 구조 추정
        # 복잡한 컴퓨터 비전 알고리즘 필요
        
        return None
    
    def _multi_view_reconstruction(self, elements: List[DiagramElement]) -> Optional[Reconstructed3DModel]:
        """다중 뷰 기반 3D 재구성"""
        # 여러 시점의 도면을 조합하여 3D 모델 생성
        # 스테레오 비전 기법 활용
        
        return None

class Patent3DProcessor:
    """특허 3D 처리 통합 클래스"""
    
    def __init__(self, db_path: str = "patent_3d_models.db"):
        self.diagram_analyzer = DiagramAnalyzer()
        self.model_converter = Diagram2ModelConverter()
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """3D 모델 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patent_3d_models (
                model_id TEXT PRIMARY KEY,
                patent_id TEXT,
                source_figures TEXT,
                model_description TEXT,
                vertices TEXT,
                faces TEXT,
                materials TEXT,
                reconstruction_method TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagram_elements (
                element_id TEXT PRIMARY KEY,
                patent_id TEXT,
                figure_path TEXT,
                element_type TEXT,
                coordinates TEXT,
                properties TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_patent_figures(self, patent_id: str, figure_paths: List[str]) -> List[Reconstructed3DModel]:
        """특허 도면들을 처리하여 3D 모델 생성"""
        
        logger.info(f"특허 {patent_id}의 {len(figure_paths)}개 도면 처리 시작")
        
        all_models = []
        
        for figure_path in figure_paths:
            try:
                # 도면 분석
                elements = self.diagram_analyzer.analyze_diagram(figure_path)
                
                # 요소들을 데이터베이스에 저장
                self._save_diagram_elements(patent_id, figure_path, elements)
                
                # 3D 모델 변환 시도 (여러 방법)
                for method in ['orthographic_projection', 'isometric_reconstruction']:
                    model = self.model_converter.convert_diagram_to_3d(elements, method)
                    
                    if model:
                        model.patent_id = patent_id
                        model.source_figures = [figure_path]
                        
                        # 모델 저장
                        self._save_3d_model(model)
                        all_models.append(model)
                        
                        logger.info(f"3D 모델 생성 성공: {model.model_id} (방법: {method})")
                        break  # 성공하면 다른 방법 시도하지 않음
                
            except Exception as e:
                logger.error(f"도면 처리 오류 ({figure_path}): {e}")
        
        logger.info(f"특허 {patent_id} 처리 완료: {len(all_models)}개 3D 모델 생성")
        return all_models
    
    def _save_diagram_elements(self, patent_id: str, figure_path: str, elements: List[DiagramElement]):
        """도면 요소들을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for element in elements:
            cursor.execute('''
                INSERT OR REPLACE INTO diagram_elements 
                (element_id, patent_id, figure_path, element_type, coordinates, properties, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{patent_id}_{element.element_id}",
                patent_id,
                figure_path,
                element.element_type,
                json.dumps(element.coordinates),
                json.dumps(element.properties),
                element.confidence
            ))
        
        conn.commit()
        conn.close()
    
    def _save_3d_model(self, model: Reconstructed3DModel):
        """3D 모델을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO patent_3d_models 
            (model_id, patent_id, source_figures, model_description, vertices, faces, 
             materials, reconstruction_method, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model.model_id,
            model.patent_id,
            json.dumps(model.source_figures),
            model.model_description,
            json.dumps(model.vertices),
            json.dumps(model.faces),
            json.dumps(model.materials),
            model.reconstruction_method,
            model.confidence_score
        ))
        
        conn.commit()
        conn.close()
    
    def export_model_to_obj(self, model_id: str, output_path: str) -> bool:
        """3D 모델을 OBJ 파일로 내보내기"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT vertices, faces FROM patent_3d_models WHERE model_id = ?', (model_id,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                logger.error(f"모델을 찾을 수 없습니다: {model_id}")
                return False
            
            vertices = json.loads(row[0])
            faces = json.loads(row[1])
            
            # OBJ 파일 작성
            with open(output_path, 'w') as f:
                f.write(f"# 3D Model exported from Patent2Tech\n")
                f.write(f"# Model ID: {model_id}\n\n")
                
                # 꼭짓점 작성
                for vertex in vertices:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                
                f.write("\n")
                
                # 면 작성 (OBJ는 1-based 인덱스 사용)
                for face in faces:
                    face_str = " ".join([str(i+1) for i in face])
                    f.write(f"f {face_str}\n")
            
            logger.info(f"OBJ 파일 내보내기 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"OBJ 내보내기 오류: {e}")
            return False
    
    def get_processing_statistics(self) -> Dict:
        """처리 통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 3D 모델 통계
        cursor.execute("SELECT COUNT(*) FROM patent_3d_models")
        total_models = cursor.fetchone()[0]
        
        cursor.execute("SELECT reconstruction_method, COUNT(*) FROM patent_3d_models GROUP BY reconstruction_method")
        method_stats = cursor.fetchall()
        
        cursor.execute("SELECT AVG(confidence_score) FROM patent_3d_models")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # 도면 요소 통계
        cursor.execute("SELECT COUNT(*) FROM diagram_elements")
        total_elements = cursor.fetchone()[0]
        
        cursor.execute("SELECT element_type, COUNT(*) FROM diagram_elements GROUP BY element_type")
        element_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_3d_models': total_models,
            'reconstruction_methods': dict(method_stats),
            'average_confidence': round(avg_confidence, 2),
            'total_diagram_elements': total_elements,
            'element_types': dict(element_stats)
        }

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🎯 특허 3D 이미지 처리 시스템")
    print("Patent 3D Image Processing System")
    print("=" * 80)
    
    # 3D 처리기 초기화
    processor = Patent3DProcessor()
    
    while True:
        print("\n📋 메뉴를 선택하세요:")
        print("1. 🔍 도면 분석 및 요소 추출")
        print("2. 🎲 3D 모델 생성")
        print("3. 📄 OBJ 파일 내보내기")
        print("4. 📊 처리 통계 조회")
        print("5. ❌ 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == '1':
            image_path = input("분석할 도면 이미지 경로를 입력하세요: ").strip()
            if os.path.exists(image_path):
                elements = processor.diagram_analyzer.analyze_diagram(image_path)
                print(f"\n✅ 도면 분석 완료: {len(elements)}개 요소 검출")
                
                for element in elements[:5]:  # 상위 5개만 표시
                    print(f"   - {element.element_type}: 신뢰도 {element.confidence:.2f}")
            else:
                print("❌ 파일을 찾을 수 없습니다.")
        
        elif choice == '2':
            patent_id = input("특허 ID를 입력하세요: ").strip()
            figure_paths = input("도면 파일 경로들을 입력하세요 (쉼표로 구분): ").strip().split(',')
            figure_paths = [path.strip() for path in figure_paths if path.strip()]
            
            if figure_paths:
                models = processor.process_patent_figures(patent_id, figure_paths)
                print(f"\n✅ 3D 모델 생성 완료: {len(models)}개")
                
                for model in models:
                    print(f"   - {model.model_id}: {model.reconstruction_method} (신뢰도: {model.confidence_score:.2f})")
            else:
                print("❌ 올바른 파일 경로를 입력해주세요.")
        
        elif choice == '3':
            model_id = input("내보낼 모델 ID를 입력하세요: ").strip()
            output_path = input("출력 파일 경로를 입력하세요 (.obj): ").strip()
            
            if processor.export_model_to_obj(model_id, output_path):
                print(f"✅ OBJ 파일 내보내기 완료: {output_path}")
            else:
                print("❌ 내보내기 실패")
        
        elif choice == '4':
            stats = processor.get_processing_statistics()
            print(f"\n📊 처리 통계:")
            print(f"- 총 3D 모델: {stats['total_3d_models']}개")
            print(f"- 평균 신뢰도: {stats['average_confidence']}")
            print(f"- 재구성 방법별 통계: {stats['reconstruction_methods']}")
            print(f"- 총 도면 요소: {stats['total_diagram_elements']}개")
            print(f"- 요소 타입별 통계: {stats['element_types']}")
        
        elif choice == '5':
            print("시스템을 종료합니다.")
            break
        
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()

