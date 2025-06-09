import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import Dashboard from "./components/Dashboard"
import Analytics from "./components/Analytics"

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
