import { useState } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { 
  Code, 
  Download, 
  Copy, 
  Play, 
  CheckCircle, 
  FileText,
  Settings,
  Share
} from 'lucide-react'

const ImplementationViewer = ({ implementation }) => {
  const [copiedCode, setCopiedCode] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [output, setOutput] = useState('')

  // 샘플 구현 데이터 (실제로는 props로 받아올 것)
  const sampleImplementation = implementation || {
    id: 1,
    type: 'code',
    content: `# 특허 기반 이미지 분류 시스템
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class PatentImageClassifier:
    """
    특허 US1234567A 기반 이미지 분류 시스템
    핵심 알고리즘: CNN 기반 특징 추출 및 분류
    """
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self, input_shape=(224, 224, 3)):
        """모델 구조 정의"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """모델 훈련"""
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def predict(self, X_test):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
            
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

# 사용 예시
if __name__ == "__main__":
    # 데이터 로드 (예시)
    # X, y = load_data()
    
    # 분류기 초기화
    classifier = PatentImageClassifier(num_classes=10)
    
    # 모델 구축
    model = classifier.build_model()
    print("모델 구조:")
    model.summary()
    
    # 훈련 (실제 데이터가 있을 때)
    # history = classifier.train(X_train, y_train)
    
    print("특허 기반 이미지 분류 시스템이 준비되었습니다.")`,
    evaluation: {
      syntax_check: 1.0,
      completeness: 0.9,
      functionality: 0.85,
      documentation: 0.8
    },
    test_cases: [
      "기본 모델 생성 테스트",
      "훈련 데이터 검증 테스트",
      "예측 정확도 테스트"
    ],
    patent_info: {
      id: "US1234567A",
      title: "AI 기반 이미지 분류 시스템",
      core_algorithm: "CNN 기반 특징 추출 및 분류"
    }
  }

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(sampleImplementation.content)
      setCopiedCode(true)
      setTimeout(() => setCopiedCode(false), 2000)
    } catch (err) {
      console.error('복사 실패:', err)
    }
  }

  const runCode = async () => {
    setIsRunning(true)
    setOutput('코드 실행 중...')
    
    // 시뮬레이션된 코드 실행
    setTimeout(() => {
      setOutput(`실행 결과:
모델 구조:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 222, 222, 32)     896       
max_pooling2d (MaxPooling2D) (None, 111, 111, 32)     0         
conv2d_1 (Conv2D)            (None, 109, 109, 64)     18496     
max_pooling2d_1 (MaxPooling2 (None, 54, 54, 64)       0         
conv2d_2 (Conv2D)            (None, 52, 52, 64)       36928     
flatten (Flatten)            (None, 173056)            0         
dense (Dense)                (None, 64)                11075648  
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 11,132,618
Trainable params: 11,132,618
Non-trainable params: 0

특허 기반 이미지 분류 시스템이 준비되었습니다.`)
      setIsRunning(false)
    }, 2000)
  }

  const downloadCode = () => {
    const element = document.createElement('a')
    const file = new Blob([sampleImplementation.content], { type: 'text/plain' })
    element.href = URL.createObjectURL(file)
    element.download = `patent_${sampleImplementation.patent_info.id}_implementation.py`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  const getEvaluationColor = (score) => {
    if (score >= 0.8) return 'bg-green-500'
    if (score >= 0.6) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">구현 상세 보기</h2>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm">
            <Share className="h-4 w-4 mr-2" />
            공유
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            설정
          </Button>
        </div>
      </div>

      {/* 특허 정보 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <FileText className="h-5 w-5" />
            <span>특허 정보</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-500">특허 ID</label>
              <p className="font-mono">{sampleImplementation.patent_info.id}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">특허 제목</label>
              <p>{sampleImplementation.patent_info.title}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">핵심 알고리즘</label>
              <p>{sampleImplementation.patent_info.core_algorithm}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 메인 컨텐츠 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 코드 영역 */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <Code className="h-5 w-5" />
                  <span>구현 코드</span>
                  <Badge variant="secondary">{sampleImplementation.type}</Badge>
                </CardTitle>
                
                <div className="flex space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={copyToClipboard}
                  >
                    {copiedCode ? (
                      <CheckCircle className="h-4 w-4" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadCode}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="default"
                    size="sm"
                    onClick={runCode}
                    disabled={isRunning}
                  >
                    <Play className="h-4 w-4 mr-1" />
                    {isRunning ? '실행중...' : '실행'}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code>{sampleImplementation.content}</code>
              </pre>
            </CardContent>
          </Card>

          {/* 실행 결과 */}
          {output && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle className="text-base">실행 결과</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
                  {output}
                </pre>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 사이드바 */}
        <div className="space-y-6">
          {/* 평가 점수 */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">구현 평가</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(sampleImplementation.evaluation).map(([metric, score]) => (
                  <div key={metric} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="capitalize">{metric.replace('_', ' ')}</span>
                      <span>{(score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${getEvaluationColor(score)}`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-4 p-3 bg-blue-50 rounded">
                <div className="text-sm font-medium text-blue-900">
                  전체 점수: {(Object.values(sampleImplementation.evaluation).reduce((a, b) => a + b, 0) / Object.values(sampleImplementation.evaluation).length * 100).toFixed(0)}%
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 테스트 케이스 */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">테스트 케이스</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {sampleImplementation.test_cases.map((testCase, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>{testCase}</span>
                  </div>
                ))}
              </div>
              
              <Button variant="outline" size="sm" className="w-full mt-3">
                테스트 실행
              </Button>
            </CardContent>
          </Card>

          {/* 추가 작업 */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">추가 작업</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Button variant="outline" size="sm" className="w-full">
                  다른 구현 생성
                </Button>
                <Button variant="outline" size="sm" className="w-full">
                  최적화 제안
                </Button>
                <Button variant="outline" size="sm" className="w-full">
                  문서화 생성
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default ImplementationViewer

