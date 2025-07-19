import { useState } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Checkbox } from '@/components/ui/checkbox.jsx'
import { Search, Loader2, Lightbulb, Code, Cpu } from 'lucide-react'

const PatentSearch = ({ onSearchResults, isLoading, setIsLoading }) => {
  const [query, setQuery] = useState('')
  const [implementationTypes, setImplementationTypes] = useState({
    code: true,
    circuit: false,
    cad: false
  })

  const handleSearch = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    
    try {
      const selectedTypes = Object.keys(implementationTypes).filter(
        type => implementationTypes[type]
      )

      const response = await fetch('http://localhost:5001/api/patent/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          implementation_types: selectedTypes
        })
      })

      const data = await response.json()
      
      if (data.success) {
        onSearchResults(data)
      } else {
        console.error('검색 실패:', data.error)
        alert('검색 중 오류가 발생했습니다: ' + data.error)
      }
    } catch (error) {
      console.error('검색 오류:', error)
      alert('검색 중 오류가 발생했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleImplementationTypeChange = (type, checked) => {
    setImplementationTypes(prev => ({
      ...prev,
      [type]: checked
    }))
  }

  const exampleQueries = [
    "인공지능 기반 이미지 인식 시스템",
    "블록체인 기반 데이터 보안 방법",
    "IoT 센서 네트워크 최적화",
    "머신러닝 추천 알고리즘",
    "자율주행 차량 제어 시스템"
  ]

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Search className="h-5 w-5" />
          <span>특허 기술 검색 및 구현 생성</span>
        </CardTitle>
        <CardDescription>
          원하는 기술이나 특허 내용을 입력하면 관련 특허를 찾아 구현 가능한 코드로 변환해드립니다.
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* 검색 입력 */}
        <div className="space-y-2">
          <label className="text-sm font-medium">검색 쿼리</label>
          <Textarea
            placeholder="예: 머신러닝을 활용한 이미지 분류 시스템"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="min-h-[100px]"
          />
        </div>

        {/* 구현 타입 선택 */}
        <div className="space-y-3">
          <label className="text-sm font-medium">생성할 구현 타입</label>
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="code"
                checked={implementationTypes.code}
                onCheckedChange={(checked) => handleImplementationTypeChange('code', checked)}
              />
              <label htmlFor="code" className="flex items-center space-x-2 text-sm">
                <Code className="h-4 w-4" />
                <span>Python 코드</span>
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Checkbox
                id="circuit"
                checked={implementationTypes.circuit}
                onCheckedChange={(checked) => handleImplementationTypeChange('circuit', checked)}
              />
              <label htmlFor="circuit" className="flex items-center space-x-2 text-sm">
                <Cpu className="h-4 w-4" />
                <span>회로 설계</span>
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Checkbox
                id="cad"
                checked={implementationTypes.cad}
                onCheckedChange={(checked) => handleImplementationTypeChange('cad', checked)}
              />
              <label htmlFor="cad" className="flex items-center space-x-2 text-sm">
                <Lightbulb className="h-4 w-4" />
                <span>CAD 설계</span>
              </label>
            </div>
          </div>
        </div>

        {/* 예시 쿼리 */}
        <div className="space-y-3">
          <label className="text-sm font-medium">예시 검색어</label>
          <div className="flex flex-wrap gap-2">
            {exampleQueries.map((example, index) => (
              <Badge
                key={index}
                variant="outline"
                className="cursor-pointer hover:bg-blue-50"
                onClick={() => setQuery(example)}
              >
                {example}
              </Badge>
            ))}
          </div>
        </div>

        {/* 검색 버튼 */}
        <Button
          onClick={handleSearch}
          disabled={!query.trim() || isLoading}
          className="w-full"
          size="lg"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              특허 검색 및 분석 중...
            </>
          ) : (
            <>
              <Search className="mr-2 h-4 w-4" />
              특허 검색 및 구현 생성
            </>
          )}
        </Button>

        {/* 도움말 */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">💡 검색 팁</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• 구체적인 기술 분야나 방법을 명시하면 더 정확한 결과를 얻을 수 있습니다</li>
            <li>• 여러 구현 타입을 선택하면 다양한 형태의 결과물을 받을 수 있습니다</li>
            <li>• 검색 결과는 실제 특허 데이터베이스에서 가져와 분석됩니다</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}

export default PatentSearch

