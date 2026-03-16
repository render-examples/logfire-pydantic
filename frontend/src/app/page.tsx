'use client'

import { useState } from 'react'
import QuestionInput from '@/components/QuestionInput'
import ProgressTracker from '@/components/ProgressTracker'
import AnswerDisplay from '@/components/AnswerDisplay'
import MetricsPanel from '@/components/MetricsPanel'
import MetricsSkeleton from '@/components/MetricsSkeleton'
import HistoryPanel from '@/components/HistoryPanel'
import { askQuestion } from '@/lib/api'
import type { AnswerResponse, ProgressUpdate } from '@/types'

export default function Home() {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<ProgressUpdate[]>([])
  const [answer, setAnswer] = useState<AnswerResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)

  const handleAskQuestion = async (question: string) => {
    setLoading(true)
    setProgress([])
    setAnswer(null)
    setError(null)

    try {
      const result = await askQuestion(question, (update) => {
        setProgress((prev) => [...prev, update])
      })

      setAnswer(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleLoadSession = (session: AnswerResponse) => {
    setAnswer(session)
    setProgress([]) // Clear progress since this is a loaded session
    setError(null)
  }

  return (
    <div className="min-h-screen bg-black flex flex-col">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-black flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="inline-flex gap-2 items-end">  
                <h1 className="text-3xl font-light font-roobert text-white">
                Render Q&A
              </h1> 
              <h3 className="mt-1 text-xl text-purple-400">
                 - hosted on Render, AI observability by Logfire
              </h3>
              </div>
              <p className="mt-1 text-sm text-zinc-400">
                Observable RAG pipeline • Multi-query retrieval • Claims verification
              </p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setHistoryOpen(!historyOpen)}
                className="text-sm text-zinc-400 hover:text-purple-600 transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                History
              </button>
              <a
                href="https://logfire.pydantic.dev/docs/why/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-zinc-400 hover:text-purple-600 transition-colors flex items-center gap-1.5"
              >
                Logfire
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
              <a
                href="https://render.com/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-zinc-400 hover:text-purple-600 transition-colors flex items-center gap-1.5"
              >
                Render Docs
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - Always 2-column layout */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full">
          {/* Left Column - Input & Answer */}
          <div className="lg:col-span-2 space-y-6">
            <QuestionInput onSubmit={handleAskQuestion} loading={loading} />
            
            {error && (
              <div className="bg-red-500/10 border border-red-500/50 p-4">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {loading && (
              <ProgressTracker progress={progress} loading={loading} />
            )}

            {answer && <AnswerDisplay answer={answer} />}
          </div>

          {/* Right Column - Metrics */}
          <div className="lg:col-span-1">
            <div className="lg:sticky lg:top-8">
              {answer ? (
                <MetricsPanel answer={answer} />
              ) : (
                <MetricsSkeleton loading={loading} />
              )}
            </div>
          </div>
        </div>
      </main>

      {/* History Panel */}
      <HistoryPanel 
        onLoadSession={handleLoadSession}
        isOpen={historyOpen}
        onToggle={() => setHistoryOpen(!historyOpen)}
      />

      {/* Footer */}
      <footer className="border-t border-zinc-800 bg-black flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-zinc-400">
            <p>
              Built with{' '}
              <a href="https://www.pydantic.dev/logfire" className="text-purple-600 hover:text-purple-500">
                Logfire
              </a>
              {' '}•{' '}
              <a href="https://render.com" className="text-purple-600 hover:text-purple-500">
                Render
              </a>
              {' '}•{' '}
              <a href="https://python.langchain.com" className="text-purple-600 hover:text-purple-500">
                LangChain
              </a>
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

