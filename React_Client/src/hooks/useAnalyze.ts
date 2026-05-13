import { useState } from 'react'
import { analyzeImage } from '../api/analyze'
import type { AnalyzeResponse, Mode } from '../types/models'

interface State {
  loading: boolean
  result: AnalyzeResponse | null
  error: string | null
}

export function useAnalyze() {
  const [state, setState] = useState<State>({ loading: false, result: null, error: null })

  async function run(file: File, mode: Mode) {
    setState({ loading: true, result: null, error: null })
    try {
      const result = await analyzeImage(file, mode)
      if (result.status === 'error') {
        setState({ loading: false, result: null, error: result.message ?? 'Unknown error' })
      } else {
        setState({ loading: false, result, error: null })
      }
    } catch (e) {
      setState({ loading: false, result: null, error: (e as Error).message })
    }
  }

  return { ...state, run }
}
