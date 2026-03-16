'use client'

import { useState, FormEvent } from 'react'

interface QuestionInputProps {
  onSubmit: (question: string) => void
  loading: boolean
}

export default function QuestionInput({ onSubmit, loading }: QuestionInputProps) {
  const [question, setQuestion] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (question.trim() && !loading) {
      onSubmit(question.trim())
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (question.trim() && !loading) {
        onSubmit(question.trim())
      }
    }
  }

  const exampleQuestions = [
    "How do I deploy a Node.js app on Render?",
    "What database plans does Render offer?",
    "How do I set up environment variables?",
    "How does autoscaling work on Render?",
  ]

  return (
    <div className="bg-black border border-zinc-800 p-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="question" className="block text-sm font-medium text-zinc-300 mb-2">
            Ask a question about Render
          </label>
          <textarea
            id="question"
            rows={4}
            className="w-full px-4 py-3 bg-zinc-900 border border-zinc-700 text-white placeholder-zinc-500 focus:outline-none focus:border-purple-600 resize-none"
            placeholder="e.g., How do I deploy a Python FastAPI app on Render?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <p className="mt-2 text-xs text-zinc-500 flex justify-between">
            <span>{question.length}/500 characters</span>
            <span className="text-zinc-600">Press Enter to submit, Shift+Enter for new line</span>
          </p>
        </div>

        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="w-full px-6 py-3 bg-purple-600 text-white font-medium border border-purple-600 hover:bg-purple-700 hover:border-purple-700 focus:outline-none focus:border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
          ) : (
            'Ask Question'
          )}
        </button>
      </form>

      <div className="mt-6 pt-6 border-t border-zinc-800">
        <p className="text-sm text-zinc-400 mb-3">Try these examples:</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {exampleQuestions.map((exampleQ, idx) => (
            <button
              key={idx}
              onClick={() => setQuestion(exampleQ)}
              disabled={loading}
              className="text-left px-3 py-2 text-sm text-zinc-300 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 hover:border-yellow-500 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {exampleQ}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

