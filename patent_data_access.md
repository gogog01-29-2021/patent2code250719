
## 특허 데이터 접근 방법

### 1. USPTO (미국 특허청)
- **Open Data Portal (ODP)**: `https://data.uspto.gov/`
  - USPTO 데이터를 무료로 검색하고 추출할 수 있는 플랫폼.
  - **Bulk Data Directory**: `https://data.uspto.gov/bulkdata` 에서 대량 데이터를 다운로드할 수 있음.
- **Developer Hub**: `https://developer.uspto.gov/`
  - API를 통해 특허 할당 정보 등을 검색할 수 있음.
- **Patent Public Search**: `https://www.uspto.gov/patents/search/patent-public-search` 웹 기반 특허 검색 도구.

### 2. KIPO (특허청)
- **KIPRIS (Korea Intellectual Property Rights Information Service)**: `https://www.kipris.or.kr/khome/info.do?page=intro`
  - 국내외 산업 재산권 정보를 무료로 검색할 수 있는 종합 서비스.
  - 영어 초록 및 법적 상태 정보도 제공.
- **KIPO 공식 웹사이트**: `https://www.kipo.go.kr/en` 에서도 정보 검색 가능.

### 3. EPO (유럽 특허청)
- **Data**: `https://www.epo.org/en/searching-for-patents/data`
  - EPO 특허 정보를 연결된 오픈 데이터, 대량 데이터 세트 또는 웹 서비스를 통해 직접 얻을 수 있음.
- **Open Patent Services (OPS)**: `https://www.epo.org/en/searching-for-patents/data/web-services/ops`
  - RESTful 아키텍처를 통해 표준화된 XML 인터페이스를 통해 EPO 데이터에 접근할 수 있는 웹 서비스.
- **Espacenet**: `https://worldwide.espacenet.com/`
  - 전 세계 수백만 건의 특허 문서를 무료로 검색할 수 있는 도구.

이러한 소스들을 활용하여 특허 데이터를 수집하는 파이프라인을 구축할 수 있습니다. 특히 대량 데이터 다운로드 및 API 접근을 통해 자동화된 수집 시스템을 고려해야 합니다.




## 특허-논문/구현 매핑 데이터 및 방법론

### 1. 매핑 데이터셋
- **USPTO AI Patent Dataset (AIPD)**: AI 관련 특허를 식별하고 매핑하는 데 사용될 수 있습니다.
- **PatentsView**: 특허 연구 데이터셋을 제공하며, 벌크 다운로드가 가능합니다.
- **DeepPatent Dataset**: USPTO에서 수집한 35만 개 이상의 특허 도면으로 구성된 데이터셋입니다.
- **Harvard USPTO Patent Dataset (HUPD)**: 대규모의 잘 구조화된 영어 특허 애플리케이션 코퍼스입니다.
- **arXiv 논문**: 최근 연구에서는 AI 혁신을 학술 연구에서 산업 특허로 매핑하는 글로벌 데이터셋에 대한 논문도 발표되었습니다.
- **The Lens**: 학술 및 특허 기록을 통합하여 분석할 수 있는 플랫폼입니다.

### 2. 매핑 방법론
- **특허-제품/부품 매핑**: 특허의 클레임(claims)을 특정 제품이나 부품에 연결하는 방법론이 제안됩니다.
- **키워드 기반 특허 맵**: 특허에서 키워드를 추출하여 기술 트렌드 및 관련성을 파악하는 데 사용될 수 있습니다.
- **멀티모달 데이터 활용**: `pasted_content.txt`에서 언급된 Pix2Struct + GPT-Vision과 같은 기술을 활용하여 특허 도면을 분석하고 기술 개념을 재구축하는 접근 방식이 유망합니다.
- **표준-특허 매핑**: 표준과 기술적으로 관련된 특허를 매핑하여 기술 혁신 연구에 활용하는 방법도 있습니다.

이러한 데이터셋과 방법론을 조합하여 특허 LLM 구축에 필요한 특허-논문/구현 매핑 데이터를 확보하고 활용할 수 있습니다. 특히, 특허 도면 분석을 통한 기술 개념 재구축은 Patent2Tech 시스템의 핵심 기능이 될 것입니다.




## 특허 관련 3D 이미지 데이터 수집 및 활용 방안

### 1. 3D 이미지 데이터 수집
- **특허청 데이터베이스**: USPTO, EPO, KIPO 등에서 제공하는 특허 문서(PDF, XML) 내의 도면을 활용합니다. 이 도면들은 2D 형태이지만, 음영 및 원근법을 통해 3D 특징을 나타낼 수 있습니다. 이를 이미지 처리 기술(OCR, Diagram Parsing)을 통해 분석하여 3D 정보를 역추론할 수 있습니다.
- **보조 저장소**: 일부 특허는 CAD 파일 또는 보조 3D 모델을 기업 또는 표준 웹사이트에 연결할 수 있습니다. 이러한 외부 소스를 탐색하여 3D 데이터를 확보합니다.
- **공개 데이터셋**: 'PatentFigure' (image2structure 작업용), 'OpenPatentImages' 또는 arXiv/IEEE의 연구 컬렉션과 같은 공개 데이터셋을 활용하여 3D 관련 이미지를 수집합니다.
- **수동 추출 및 재구성**: 3D 파일이 존재하지 않는 경우, `diagram2model` 알고리즘(예: Pix2Struct, OCR+SVG+CAD 재구성)을 사용하여 2D 도면에서 3D 모델을 재구성하는 방법을 고려합니다.

### 2. 3D 이미지 데이터 활용
- **기술 개념 재구성**: 추출된 3D 도면은 기술 클레임(claims)과 매핑되어(예: Pix2Struct + Vision-Language Models 활용) 구현 로직을 재구성하는 데 도움을 줍니다. 이는 코드 생성(Python 모듈, CAD, 회로 넷리스트)에 활용될 수 있습니다.
- **시뮬레이션 및 시각화**: 생성된 3D 자산은 OpenSCAD, Fusion360과 같은 도구에서 시각화하거나 시뮬레이션에 활용될 수 있습니다.
- **선행 기술 유사성 평가**: 3D 모델은 특허 검색 및 법적 평가(선행 기술 유사성)를 위한 추가적인 특징을 제공할 수 있습니다.

3D 이미지 데이터는 Patent2Tech 시스템에서 특허의 기술적 내용을 심층적으로 이해하고, 이를 실제 구현 가능한 형태로 변환하는 데 중요한 역할을 할 것입니다.

