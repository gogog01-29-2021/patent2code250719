import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, Clock, CheckCircle, XCircle, Code, Cpu, Lightbulb } from 'lucide-react'

const Statistics = () => {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStatistics()
  }, [])

  const fetchStatistics = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/patent/statistics')
      const data = await response.json()
      
      if (data.success) {
        setStats(data.statistics)
      }
    } catch (error) {
      console.error('통계 조회 오류:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (!stats) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-center text-gray-500">통계 데이터를 불러올 수 없습니다.</p>
        </CardContent>
      </Card>
    )
  }

  // 구현 타입별 데이터 준비
  const implementationData = Object.entries(stats.implementation_types || {}).map(([type, count]) => ({
    name: type === 'code' ? 'Python 코드' : type === 'circuit' ? '회로 설계' : type,
    value: count,
    icon: type === 'code' ? Code : type === 'circuit' ? Cpu : Lightbulb
  }))

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">시스템 통계</h2>
        <Badge variant="outline">실시간 데이터</Badge>
      </div>

      {/* 주요 지표 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">총 쿼리 수</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_queries}</div>
            <p className="text-xs text-muted-foreground">
              누적 검색 요청
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">성공률</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.success_rate.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              {stats.successful_queries}/{stats.total_queries} 성공
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">분석된 특허</CardTitle>
            <Code className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_analyses}</div>
            <p className="text-xs text-muted-foreground">
              기술 개념 분석 완료
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">평균 처리 시간</CardTitle>
            <Clock className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.average_processing_time}초</div>
            <p className="text-xs text-muted-foreground">
              최근 10개 쿼리 기준
            </p>
          </CardContent>
        </Card>
      </div>

      {/* 차트 섹션 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 구현 타입별 분포 */}
        <Card>
          <CardHeader>
            <CardTitle>구현 타입별 분포</CardTitle>
            <CardDescription>
              생성된 구현물의 타입별 현황
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={implementationData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {implementationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* 구현 타입별 막대 차트 */}
        <Card>
          <CardHeader>
            <CardTitle>구현 생성 현황</CardTitle>
            <CardDescription>
              총 {stats.total_implementations}개의 구현물이 생성되었습니다
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={implementationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* 상세 통계 */}
      <Card>
        <CardHeader>
          <CardTitle>상세 통계</CardTitle>
          <CardDescription>
            시스템 성능 및 사용 현황에 대한 자세한 정보
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium">처리 성능</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">평균 처리 시간</span>
                  <span className="text-sm font-medium">{stats.average_processing_time}초</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">성공률</span>
                  <span className="text-sm font-medium">{stats.success_rate.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">실패한 쿼리</span>
                  <span className="text-sm font-medium">{stats.total_queries - stats.successful_queries}개</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-medium">생성 현황</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">총 분석 수</span>
                  <span className="text-sm font-medium">{stats.total_analyses}개</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">총 구현 수</span>
                  <span className="text-sm font-medium">{stats.total_implementations}개</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">분석당 평균 구현</span>
                  <span className="text-sm font-medium">
                    {stats.total_analyses > 0 ? (stats.total_implementations / stats.total_analyses).toFixed(1) : 0}개
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default Statistics

