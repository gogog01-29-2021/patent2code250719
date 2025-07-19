import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Search, FileText, Code, Zap, BarChart3, History, Settings } from 'lucide-react'
import './App.css'

// 컴포넌트들
import PatentSearch from './components/PatentSearch'
import AnalysisResults from './components/AnalysisResults'
import ImplementationViewer from './components/ImplementationViewer'
import Statistics from './components/Statistics'
import QueryHistory from './components/QueryHistory'

function App() {
  const [currentView, setCurrentView] = useState('search')
  const [searchResults, setSearchResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* 헤더 */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Zap className="h-8 w-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Patent2Tech</h1>
              </div>
              <Badge variant="secondary" className="text-xs">
                LLM 기반 특허 구현 시스템
              </Badge>
            </div>
            
            <nav className="flex space-x-4">
              <Button
                variant={currentView === 'search' ? 'default' : 'ghost'}
                onClick={() => setCurrentView('search')}
                className="flex items-center space-x-2"
              >
                <Search className="h-4 w-4" />
                <span>검색</span>
              </Button>
              <Button
                variant={currentView === 'history' ? 'default' : 'ghost'}
                onClick={() => setCurrentView('history')}
                className="flex items-center space-x-2"
              >
                <History className="h-4 w-4" />
                <span>기록</span>
              </Button>
              <Button
                variant={currentView === 'statistics' ? 'default' : 'ghost'}
                onClick={() => setCurrentView('statistics')}
                className="flex items-center space-x-2"
              >
                <BarChart3 className="h-4 w-4" />
                <span>통계</span>
              </Button>
            </nav>
          </div>
        </div>
      </header>

      {/* 메인 컨텐츠 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentView === 'search' && (
          <div className="space-y-8">
            {/* 검색 섹션 */}
            <PatentSearch 
              onSearchResults={setSearchResults}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
            
            {/* 결과 섹션 */}
            {searchResults && (
              <AnalysisResults 
                results={searchResults}
                onViewImplementation={(impl) => setCurrentView('implementation')}
              />
            )}
          </div>
        )}

        {currentView === 'history' && <QueryHistory />}
        {currentView === 'statistics' && <Statistics />}
        {currentView === 'implementation' && <ImplementationViewer />}
      </main>

      {/* 푸터 */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p className="text-sm">
              Patent2Tech LLM System - 특허에서 기술 구현으로
            </p>
            <p className="text-xs mt-2">
              Powered by AI • Built with React & Flask
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App

