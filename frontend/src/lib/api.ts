import type { AnswerResponse, ProgressUpdate } from '../types'

// In production, NEXT_PUBLIC_API_URL will be the Render service hostname
// In development, it falls back to localhost
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL 
  ? `https://${process.env.NEXT_PUBLIC_API_URL}.onrender.com` 
  : 'http://localhost:8000'

export async function askQuestion(
  question: string,
  onProgress?: (update: ProgressUpdate) => void
): Promise<AnswerResponse> {
  if (onProgress) {
    // Use streaming endpoint
    return askQuestionStream(question, onProgress)
  } else {
    // Use regular endpoint
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  }
}

async function askQuestionStream(
  question: string,
  onProgress: (update: ProgressUpdate) => void
): Promise<AnswerResponse> {
  const response = await fetch(`${API_BASE_URL}/ask/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('No response body')
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult: AnswerResponse | null = null

  while (true) {
    const { done, value } = await reader.read()
    
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6))
        
        if (data.type === 'complete') {
          finalResult = data.result
        } else if (data.type === 'error') {
          throw new Error(data.message)
        } else {
          // Progress update
          onProgress(data as ProgressUpdate)
        }
      }
    }
  }

  if (!finalResult) {
    throw new Error('No final result received')
  }

  return finalResult
}

export async function checkHealth(): Promise<{ status: string; database_connected: boolean }> {
  const response = await fetch(`${API_BASE_URL}/health`)
  
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`)
  }

  return response.json()
}

export interface HistorySession {
  id: string
  question: string
  answer: string
  sources: any[]
  claims: any[]
  evaluations: any[]
  quality_score: number
  iterations: number
  total_cost: number
  total_duration_ms: number
  created_at: string
  stages?: any[]  // Pipeline stages (optional for backwards compatibility)
  trace_id?: string  // Logfire trace ID (optional)
}

export async function getHistory(limit: number = 20): Promise<HistorySession[]> {
  const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`)
  
  if (!response.ok) {
    throw new Error(`Failed to fetch history: ${response.statusText}`)
  }

  const data = await response.json()
  return data.sessions
}

export async function getSession(sessionId: string): Promise<HistorySession> {
  const response = await fetch(`${API_BASE_URL}/history/${sessionId}`)
  
  if (!response.ok) {
    throw new Error(`Failed to fetch session: ${response.statusText}`)
  }

  return response.json()
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/history/${sessionId}`, {
    method: 'DELETE',
  })
  
  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`)
  }
}

export async function clearAllHistory(): Promise<{ count: number }> {
  const response = await fetch(`${API_BASE_URL}/history`, {
    method: 'DELETE',
  })
  
  if (!response.ok) {
    throw new Error(`Failed to clear history: ${response.statusText}`)
  }

  return response.json()
}

export interface LogfireLog {
  timestamp: string
  message: string
  level: string
  span_name: string
  attributes: Record<string, any>
  service_name: string
}

export interface SessionLogsResponse {
  trace_id: string
  logs: LogfireLog[]
  columns: string[]
}

export async function getSessionLogs(sessionId: string): Promise<SessionLogsResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/logs`)
  
  if (!response.ok) {
    if (response.status === 404) {
      const error = await response.json()
      throw new Error(error.detail || 'Logs not found')
    }
    if (response.status === 501) {
      throw new Error('Logfire integration not configured')
    }
    throw new Error('Failed to fetch logs')
  }
  
  return response.json()
}

