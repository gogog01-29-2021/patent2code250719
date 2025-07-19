#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특허 LLM 훈련 및 파인튜닝 시스템
Patent LLM Training and Fine-tuning System

특허 데이터를 활용한 LLM 파인튜닝 및 훈련 관리 시스템

MIT License
Copyright (c) 2025
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """훈련 설정 클래스"""
    base_model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "patent_llm_output"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    dataloader_num_workers: int = 4

@dataclass
class TrainingExample:
    """훈련 예제 데이터 클래스"""
    input_text: str
    target_text: str
    patent_id: str
    example_type: str  # "claim_analysis", "implementation_generation", "evaluation"
    metadata: Dict[str, Any]

class PatentTrainingDataset(Dataset):
    """특허 훈련 데이터셋"""
    
    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 입력과 타겟을 결합하여 하나의 텍스트로 만듦
        full_text = f"{example.input_text} {self.tokenizer.eos_token} {example.target_text}"
        
        # 토크나이징
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # 라벨은 입력과 동일 (언어 모델링)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class PatentDatasetBuilder:
    """특허 훈련 데이터셋 구축기"""
    
    def __init__(self, processed_patents_db: str = "processed_patents.db"):
        self.db_path = processed_patents_db
        self.prompt_templates = {
            "claim_analysis": {
                "input": "다음 특허 클레임을 분석하여 핵심 기술 개념을 추출해주세요:\n\n특허 제목: {title}\n클레임: {claim}\n\n분석:",
                "target": "핵심 알고리즘: {core_algorithm}\n기술 구성요소: {components}\n구현 요구사항: {requirements}"
            },
            "implementation_generation": {
                "input": "다음 기술 개념을 바탕으로 Python 구현 코드를 생성해주세요:\n\n기술 개념: {concept}\n구성요소: {components}\n\n구현:",
                "target": "{implementation_code}"
            },
            "evaluation": {
                "input": "다음 구현 코드를 평가해주세요:\n\n코드:\n{code}\n\n평가:",
                "target": "구문 정확성: {syntax_score}\n완성도: {completeness_score}\n기능성: {functionality_score}\n전체 평가: {overall_assessment}"
            }
        }
    
    def build_training_dataset(self, max_examples: int = 10000) -> List[TrainingExample]:
        """훈련 데이터셋 구축"""
        
        logger.info("특허 훈련 데이터셋 구축 시작")
        
        examples = []
        
        # 데이터베이스에서 처리된 특허 데이터 로드
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT patent_id, processed_text, structured_claims, technical_keywords
            FROM processed_patents
            LIMIT ?
        ''', (max_examples,))
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            patent_id, processed_text, structured_claims_json, keywords_json = row
            
            try:
                structured_claims = json.loads(structured_claims_json) if structured_claims_json else []
                keywords = json.loads(keywords_json) if keywords_json else []
                
                # 클레임 분석 예제 생성
                examples.extend(self._create_claim_analysis_examples(patent_id, structured_claims, keywords))
                
                # 구현 생성 예제 생성
                examples.extend(self._create_implementation_examples(patent_id, structured_claims, keywords))
                
                # 평가 예제 생성
                examples.extend(self._create_evaluation_examples(patent_id, structured_claims))
                
            except Exception as e:
                logger.error(f"데이터셋 구축 오류 (특허 {patent_id}): {e}")
        
        logger.info(f"훈련 데이터셋 구축 완료: {len(examples)}개 예제")
        return examples
    
    def _create_claim_analysis_examples(self, patent_id: str, claims: List[Dict], keywords: List[str]) -> List[TrainingExample]:
        """클레임 분석 예제 생성"""
        examples = []
        
        for claim in claims[:3]:  # 상위 3개 클레임만 사용
            input_text = self.prompt_templates["claim_analysis"]["input"].format(
                title=f"Patent {patent_id}",
                claim=claim.get("content", "")[:500]  # 길이 제한
            )
            
            target_text = self.prompt_templates["claim_analysis"]["target"].format(
                core_algorithm=f"분석된 알고리즘 (클레임 {claim.get('claim_number', 1)})",
                components=", ".join(claim.get("technical_elements", [])[:3]),
                requirements=f"구현 요구사항 ({claim.get('type', 'unknown')} 타입)"
            )
            
            example = TrainingExample(
                input_text=input_text,
                target_text=target_text,
                patent_id=patent_id,
                example_type="claim_analysis",
                metadata={"claim_number": claim.get("claim_number", 1)}
            )
            examples.append(example)
        
        return examples
    
    def _create_implementation_examples(self, patent_id: str, claims: List[Dict], keywords: List[str]) -> List[TrainingExample]:
        """구현 생성 예제 생성"""
        examples = []
        
        if claims:
            main_claim = claims[0]  # 첫 번째 클레임 사용
            
            input_text = self.prompt_templates["implementation_generation"]["input"].format(
                concept=main_claim.get("content", "")[:300],
                components=", ".join(main_claim.get("technical_elements", [])[:3])
            )
            
            # 간단한 구현 코드 템플릿 생성
            implementation_code = self._generate_sample_implementation(main_claim, keywords)
            
            target_text = self.prompt_templates["implementation_generation"]["target"].format(
                implementation_code=implementation_code
            )
            
            example = TrainingExample(
                input_text=input_text,
                target_text=target_text,
                patent_id=patent_id,
                example_type="implementation_generation",
                metadata={"claim_type": main_claim.get("type", "unknown")}
            )
            examples.append(example)
        
        return examples
    
    def _create_evaluation_examples(self, patent_id: str, claims: List[Dict]) -> List[TrainingExample]:
        """평가 예제 생성"""
        examples = []
        
        # 샘플 코드와 평가 생성
        sample_code = """
def patent_algorithm():
    # 특허 알고리즘 구현
    result = process_input()
    return result

def process_input():
    return "processed"
"""
        
        input_text = self.prompt_templates["evaluation"]["input"].format(code=sample_code)
        
        target_text = self.prompt_templates["evaluation"]["target"].format(
            syntax_score="9/10",
            completeness_score="7/10",
            functionality_score="6/10",
            overall_assessment="기본적인 구조는 갖추었으나 구체적인 구현이 필요함"
        )
        
        example = TrainingExample(
            input_text=input_text,
            target_text=target_text,
            patent_id=patent_id,
            example_type="evaluation",
            metadata={}
        )
        examples.append(example)
        
        return examples
    
    def _generate_sample_implementation(self, claim: Dict, keywords: List[str]) -> str:
        """샘플 구현 코드 생성"""
        claim_type = claim.get("type", "method")
        
        if claim_type == "method":
            return f"""
import numpy as np

def patent_method():
    \"\"\"
    특허 방법 구현
    클레임: {claim.get('content', '')[:100]}...
    \"\"\"
    # 입력 처리
    input_data = preprocess_input()
    
    # 핵심 알고리즘 실행
    result = core_algorithm(input_data)
    
    # 결과 후처리
    output = postprocess_result(result)
    
    return output

def preprocess_input():
    return np.array([1, 2, 3])

def core_algorithm(data):
    return data * 2

def postprocess_result(result):
    return result.tolist()
"""
        else:
            return f"""
class PatentApparatus:
    \"\"\"
    특허 장치 구현
    클레임: {claim.get('content', '')[:100]}...
    \"\"\"
    
    def __init__(self):
        self.components = []
        self.state = "initialized"
    
    def operate(self):
        self.state = "operating"
        return self.process()
    
    def process(self):
        return "processed_output"
"""

class PatentLLMTrainer:
    """특허 LLM 훈련기"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_dataset = None
        self.eval_dataset = None
        
        # 출력 디렉토리 생성
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_model_and_tokenizer(self):
        """모델 및 토크나이저 설정"""
        logger.info(f"모델 로딩: {self.config.base_model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # LoRA 설정 적용
        if self.config.use_lora:
            logger.info("LoRA 설정 적용")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def prepare_datasets(self, training_examples: List[TrainingExample], 
                        eval_split: float = 0.1):
        """데이터셋 준비"""
        logger.info("훈련 데이터셋 준비")
        
        # 훈련/평가 분할
        split_idx = int(len(training_examples) * (1 - eval_split))
        train_examples = training_examples[:split_idx]
        eval_examples = training_examples[split_idx:]
        
        # 데이터셋 생성
        self.training_dataset = PatentTrainingDataset(
            train_examples, self.tokenizer, self.config.max_length
        )
        self.eval_dataset = PatentTrainingDataset(
            eval_examples, self.tokenizer, self.config.max_length
        )
        
        logger.info(f"훈련 예제: {len(train_examples)}, 평가 예제: {len(eval_examples)}")
    
    def train(self):
        """모델 훈련"""
        if not self.model or not self.training_dataset:
            raise ValueError("모델과 데이터셋을 먼저 설정해주세요.")
        
        logger.info("특허 LLM 훈련 시작")
        
        # 훈련 인자 설정
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=["wandb"] if wandb.run else None
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 인과적 언어 모델링
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.training_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 훈련 실행
        trainer.train()
        
        # 최종 모델 저장
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"훈련 완료. 모델 저장 위치: {self.config.output_dir}")
    
    def evaluate_model(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """모델 평가"""
        if not self.model:
            raise ValueError("모델을 먼저 로드해주세요.")
        
        logger.info("모델 평가 시작")
        
        test_dataset = PatentTrainingDataset(test_examples, self.tokenizer, self.config.max_length)
        
        # 평가 메트릭 계산
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(test_dataset, batch_size=self.config.batch_size)):
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # 간단한 정확도 계산 (실제로는 더 정교한 메트릭 필요)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == batch["labels"]).sum().item()
                total_predictions += batch["labels"].numel()
        
        avg_loss = total_loss / len(test_dataset)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_perplexity": np.exp(avg_loss)
        }
        
        logger.info(f"평가 결과: {metrics}")
        return metrics

class PatentLLMManager:
    """특허 LLM 관리 시스템"""
    
    def __init__(self, models_dir: str = "patent_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = "patent_llm_training.db"
        self.init_database()
    
    def init_database(self):
        """훈련 기록 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT,
                config TEXT,
                start_time TEXT,
                end_time TEXT,
                final_metrics TEXT,
                model_path TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_training_run(self, config: TrainingConfig, dataset_size: int) -> str:
        """훈련 실행 시작"""
        run_id = f"patent_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"훈련 실행 시작: {run_id}")
        
        try:
            # 데이터셋 구축
            dataset_builder = PatentDatasetBuilder()
            training_examples = dataset_builder.build_training_dataset(dataset_size)
            
            if not training_examples:
                raise ValueError("훈련 데이터가 없습니다.")
            
            # 훈련기 초기화
            trainer = PatentLLMTrainer(config)
            trainer.setup_model_and_tokenizer()
            trainer.prepare_datasets(training_examples)
            
            # 훈련 기록 저장
            self._save_training_run(run_id, config, "running")
            
            # 훈련 실행
            trainer.train()
            
            # 평가 실행
            eval_examples = training_examples[-100:]  # 마지막 100개로 평가
            metrics = trainer.evaluate_model(eval_examples)
            
            # 모델 저장 경로
            model_path = self.models_dir / run_id
            
            # 훈련 완료 기록
            self._update_training_run(run_id, metrics, str(model_path), "completed")
            
            logger.info(f"훈련 완료: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"훈련 실행 오류: {e}")
            self._update_training_run(run_id, {}, "", "failed")
            raise
    
    def _save_training_run(self, run_id: str, config: TrainingConfig, status: str):
        """훈련 실행 기록 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_runs 
            (run_id, model_name, config, start_time, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            run_id,
            config.base_model_name,
            json.dumps(asdict(config)),
            datetime.now().isoformat(),
            status
        ))
        
        conn.commit()
        conn.close()
    
    def _update_training_run(self, run_id: str, metrics: Dict, model_path: str, status: str):
        """훈련 실행 기록 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_runs 
            SET end_time = ?, final_metrics = ?, model_path = ?, status = ?
            WHERE run_id = ?
        ''', (
            datetime.now().isoformat(),
            json.dumps(metrics),
            model_path,
            status,
            run_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_history(self) -> List[Dict]:
        """훈련 기록 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT run_id, model_name, start_time, end_time, final_metrics, status
            FROM training_runs
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            run_id, model_name, start_time, end_time, metrics_json, status = row
            
            metrics = json.loads(metrics_json) if metrics_json else {}
            
            history.append({
                "run_id": run_id,
                "model_name": model_name,
                "start_time": start_time,
                "end_time": end_time,
                "metrics": metrics,
                "status": status
            })
        
        return history

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🚀 특허 LLM 훈련 및 파인튜닝 시스템")
    print("Patent LLM Training and Fine-tuning System")
    print("=" * 80)
    
    # 관리자 초기화
    manager = PatentLLMManager()
    
    while True:
        print("\n📋 메뉴를 선택하세요:")
        print("1. 🎯 새 훈련 실행")
        print("2. 📊 훈련 기록 조회")
        print("3. ⚙️ 훈련 설정 구성")
        print("4. 📈 데이터셋 통계")
        print("5. ❌ 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == '1':
            print("\n🎯 새 훈련 실행 설정")
            
            # 기본 설정
            config = TrainingConfig()
            
            # 사용자 입력
            model_name = input(f"베이스 모델 [{config.base_model_name}]: ").strip()
            if model_name:
                config.base_model_name = model_name
            
            epochs_input = input(f"훈련 에포크 수 [{config.num_epochs}]: ").strip()
            if epochs_input.isdigit():
                config.num_epochs = int(epochs_input)
            
            dataset_size_input = input("데이터셋 크기 [1000]: ").strip()
            dataset_size = int(dataset_size_input) if dataset_size_input.isdigit() else 1000
            
            use_lora = input(f"LoRA 사용 여부 (y/n) [{'y' if config.use_lora else 'n'}]: ").strip().lower()
            if use_lora in ['y', 'n']:
                config.use_lora = use_lora == 'y'
            
            print(f"\n훈련 시작...")
            try:
                run_id = manager.start_training_run(config, dataset_size)
                print(f"✅ 훈련 완료: {run_id}")
            except Exception as e:
                print(f"❌ 훈련 실패: {e}")
        
        elif choice == '2':
            history = manager.get_training_history()
            
            if history:
                print(f"\n📊 훈련 기록 ({len(history)}개):")
                for i, run in enumerate(history[:10], 1):  # 최근 10개만 표시
                    print(f"{i}. {run['run_id']}")
                    print(f"   모델: {run['model_name']}")
                    print(f"   상태: {run['status']}")
                    print(f"   시작: {run['start_time']}")
                    if run['metrics']:
                        print(f"   평가 손실: {run['metrics'].get('eval_loss', 'N/A'):.4f}")
                    print()
            else:
                print("훈련 기록이 없습니다.")
        
        elif choice == '3':
            print("\n⚙️ 훈련 설정 구성")
            config = TrainingConfig()
            print(f"현재 설정:")
            print(f"- 베이스 모델: {config.base_model_name}")
            print(f"- 최대 길이: {config.max_length}")
            print(f"- 배치 크기: {config.batch_size}")
            print(f"- 학습률: {config.learning_rate}")
            print(f"- 에포크 수: {config.num_epochs}")
            print(f"- LoRA 사용: {config.use_lora}")
        
        elif choice == '4':
            print("\n📈 데이터셋 통계")
            try:
                builder = PatentDatasetBuilder()
                examples = builder.build_training_dataset(100)  # 샘플 100개
                
                type_counts = {}
                for example in examples:
                    type_counts[example.example_type] = type_counts.get(example.example_type, 0) + 1
                
                print(f"총 예제 수: {len(examples)}")
                print("예제 타입별 분포:")
                for example_type, count in type_counts.items():
                    print(f"  - {example_type}: {count}개")
                
            except Exception as e:
                print(f"❌ 데이터셋 통계 조회 실패: {e}")
        
        elif choice == '5':
            print("시스템을 종료합니다.")
            break
        
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()

