import type { AnswerResponse, ProgressUpdate } from '../types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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

