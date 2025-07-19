import { useState } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible.jsx'
import { 
  FileText, 
  Code, 
  Cpu, 
  ChevronDown, 
  ChevronUp, 
  Copy, 
  Download,
  Star,
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-react'

const AnalysisResults = ({ results, onViewImplementation }) => {
  const [expandedPatents, setExpandedPatents] = useState({})
  const [copiedCode, setCopiedCode] = useState({})

  const togglePatentExpansion = (patentId) => {
    setExpandedPatents(prev => ({
      ...prev,
      [patentId]: !prev[patentId]
    }))
  }

  const copyToClipboard = async (text, id) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedCode(prev => ({ ...prev, [id]: true }))
      setTimeout(() => {
        setCopiedCode(prev => ({ ...prev, [id]: false }))
      }, 2000)
    } catch (err) {
      console.error('복사 실패:', err)
    }
  }

  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  const getEvaluationColor = (score) => {
    if (score >= 0.8) return 'bg-green-500'
    if (score >= 0.6) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  if (!results) return null

  return (
    <div className="space-y-6">
      {/* 검색 요약 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>검색 결과 요약</span>
            <Badge variant="outline" className="flex items-center space-x-1">
              <Clock className="h-3 w-3" />
              <span>{results.processing_time?.toFixed(2)}초</span>
            </Badge>
          </CardTitle>
          <CardDescription>
            "{results.query}"에 대한 {results.results_count}개의 특허 분석 결과
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{results.found_patents}</div>
              <div className="text-sm text-blue-800">발견된 특허</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{results.results_count}</div>
              <div className="text-sm text-green-800">분석된 특허</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {results.raw_results?.implementations?.length || 0}
              </div>
              <div className="text-sm text-purple-800">생성된 구현</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 특허별 분석 결과 */}
      {results.raw_results?.processed_patents?.map((patent, index) => (
        <Card key={patent.patent_id} className="overflow-hidden">
          <CardHeader>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="h-5 w-5" />
                  <span className="truncate">{patent.title}</span>
                </CardTitle>
                <CardDescription className="mt-1">
                  특허 ID: {patent.patent_id}
                </CardDescription>
              </div>
              <div className="flex items-center space-x-2">
                <Badge 
                  className={`${getConfidenceColor(patent.concept.confidence_score)} border-0`}
                >
                  신뢰도 {(patent.concept.confidence_score * 100).toFixed(0)}%
                </Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => togglePatentExpansion(patent.patent_id)}
                >
                  {expandedPatents[patent.patent_id] ? 
                    <ChevronUp className="h-4 w-4" /> : 
                    <ChevronDown className="h-4 w-4" />
                  }
                </Button>
              </div>
            </div>
          </CardHeader>

          <Collapsible 
            open={expandedPatents[patent.patent_id]}
            onOpenChange={() => togglePatentExpansion(patent.patent_id)}
          >
            <CollapsibleContent>
              <CardContent className="pt-0">
                <Tabs defaultValue="concept" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="concept">기술 개념</TabsTrigger>
                    <TabsTrigger value="implementations">구현 결과</TabsTrigger>
                    <TabsTrigger value="evaluation">평가</TabsTrigger>
                  </TabsList>

                  <TabsContent value="concept" className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2">핵심 알고리즘</h4>
                      <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                        {patent.concept.core_algorithm}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-2">기술 구성요소</h4>
                      <div className="flex flex-wrap gap-2">
                        {patent.concept.technical_components.map((component, idx) => (
                          <Badge key={idx} variant="secondary">
                            {component}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="implementations" className="space-y-4">
                    {patent.implementations.map((impl, implIndex) => (
                      <Card key={implIndex} className="border-l-4 border-l-blue-500">
                        <CardHeader className="pb-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              {impl.type === 'code' && <Code className="h-4 w-4" />}
                              {impl.type === 'circuit' && <Cpu className="h-4 w-4" />}
                              <span className="font-medium capitalize">{impl.type} 구현</span>
                            </div>
                            <div className="flex space-x-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => copyToClipboard(impl.content, `${patent.patent_id}-${implIndex}`)}
                              >
                                {copiedCode[`${patent.patent_id}-${implIndex}`] ? (
                                  <CheckCircle className="h-4 w-4" />
                                ) : (
                                  <Copy className="h-4 w-4" />
                                )}
                              </Button>
                              <Button variant="outline" size="sm">
                                <Download className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                            <code>{impl.content}</code>
                          </pre>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="evaluation" className="space-y-4">
                    {patent.implementations.map((impl, implIndex) => (
                      <Card key={implIndex}>
                        <CardHeader>
                          <CardTitle className="text-base">
                            {impl.type} 구현 평가
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {Object.entries(impl.evaluation).map(([metric, score]) => (
                              <div key={metric} className="flex items-center justify-between">
                                <span className="text-sm font-medium capitalize">
                                  {metric.replace('_', ' ')}
                                </span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-24 bg-gray-200 rounded-full h-2">
                                    <div
                                      className={`h-2 rounded-full ${getEvaluationColor(score)}`}
                                      style={{ width: `${score * 100}%` }}
                                    />
                                  </div>
                                  <span className="text-sm text-gray-600 w-12">
                                    {(score * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                          
                          <div className="mt-4 p-3 bg-gray-50 rounded">
                            <div className="flex items-center space-x-2">
                              <Star className="h-4 w-4 text-yellow-500" />
                              <span className="text-sm font-medium">
                                전체 점수: {(Object.values(impl.evaluation).reduce((a, b) => a + b, 0) / Object.values(impl.evaluation).length * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </CollapsibleContent>
          </Collapsible>
        </Card>
      ))}

      {/* 추가 작업 버튼 */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="flex items-center space-x-2">
              <Download className="h-4 w-4" />
              <span>전체 결과 다운로드</span>
            </Button>
            <Button variant="outline" className="flex items-center space-x-2">
              <Code className="h-4 w-4" />
              <span>추가 구현 생성</span>
            </Button>
            <Button variant="outline" className="flex items-center space-x-2">
              <FileText className="h-4 w-4" />
              <span>보고서 생성</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default AnalysisResults

