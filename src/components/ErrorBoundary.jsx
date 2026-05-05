import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('Render error:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="h-screen dsx-bg-dark flex items-center justify-center">
          <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-8 max-w-md text-center">
            <p className="text-white font-medium mb-2">Something went wrong</p>
            <p className="text-white/60 text-sm mb-5">{this.state.error.message}</p>
            <button
              onClick={() => this.setState({ error: null })}
              className="bg-white/20 hover:bg-white/30 border border-white/20 rounded-xl px-4 py-2 text-white text-sm"
            >
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
