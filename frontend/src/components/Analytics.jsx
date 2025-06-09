import { Link } from "react-router-dom"
import { ArrowLeft, TrendingUp, TrendingDown, Activity, Zap, Droplets, Wind, Thermometer } from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts"

const Analytics = () => {
  const energyData = [
    { time: "00:00", consumption: 2.1 },
    { time: "04:00", consumption: 1.8 },
    { time: "08:00", consumption: 3.2 },
    { time: "12:00", consumption: 4.1 },
    { time: "16:00", consumption: 3.8 },
    { time: "20:00", consumption: 4.5 },
    { time: "24:00", consumption: 2.8 },
  ]

  const waterUsageData = [
    { day: "Mon", usage: 120 },
    { day: "Tue", usage: 110 },
    { day: "Wed", usage: 130 },
    { day: "Thu", usage: 115 },
    { day: "Fri", usage: 140 },
    { day: "Sat", usage: 160 },
    { day: "Sun", usage: 125 },
  ]

  const airQualityData = [
    { time: "00:00", PM25: 15, PM10: 25, CO2: 400 },
    { time: "04:00", PM25: 12, PM10: 22, CO2: 380 },
    { time: "08:00", PM25: 18, PM10: 30, CO2: 450 },
    { time: "12:00", PM25: 22, PM10: 35, CO2: 480 },
    { time: "16:00", PM25: 20, PM10: 32, CO2: 460 },
    { time: "20:00", PM25: 16, PM10: 28, CO2: 420 },
    { time: "24:00", PM25: 14, PM10: 26, CO2: 400 },
  ]

  const deviceUsageData = [
    { name: "Lighting", value: 35, color: "#06b6d4" },
    { name: "HVAC", value: 40, color: "#f59e0b" },
    { name: "Security", value: 15, color: "#10b981" },
    { name: "Others", value: 10, color: "#8b5cf6" },
  ]

  const temperatureData = [
    { time: "00:00", temp: 20.5, humidity: 48 },
    { time: "04:00", temp: 19.8, humidity: 52 },
    { time: "08:00", temp: 21.2, humidity: 46 },
    { time: "12:00", temp: 23.1, humidity: 42 },
    { time: "16:00", temp: 24.5, humidity: 38 },
    { time: "20:00", temp: 22.8, humidity: 44 },
    { time: "24:00", temp: 21.5, humidity: 47 },
  ]

  const tooltipStyle = {
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: "8px",
    color: "#fff",
  }

  return (
    <div className="min-h-screen bg-dashboard">
      {/* Header */}
      <header className="px-6 py-4 bg-slate-900/50 backdrop-blur-sm border-b border-slate-700/50">
        <div className="flex items-center space-x-4">
          <Link to="/" className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5 text-slate-400" />
          </Link>
          <div>
            <h1 className="text-white font-semibold text-lg">Analytics Dashboard</h1>
            <p className="text-slate-400 text-sm">Comprehensive system insights</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6 space-y-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="card-dark rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                <span className="text-slate-300 text-sm">Energy Usage</span>
              </div>
              <TrendingDown className="w-4 h-4 text-green-400" />
            </div>
            <div className="text-2xl font-bold text-white mb-1">3.2 kWh</div>
            <div className="text-green-400 text-sm">-12% from yesterday</div>
          </div>

          <div className="card-dark rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Droplets className="w-5 h-5 text-blue-400" />
                <span className="text-slate-300 text-sm">Water Usage</span>
              </div>
              <TrendingUp className="w-4 h-4 text-red-400" />
            </div>
            <div className="text-2xl font-bold text-white mb-1">125L</div>
            <div className="text-red-400 text-sm">+8% from yesterday</div>
          </div>

          <div className="card-dark rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Wind className="w-5 h-5 text-cyan-400" />
                <span className="text-slate-300 text-sm">Air Quality</span>
              </div>
              <Activity className="w-4 h-4 text-green-400" />
            </div>
            <div className="text-2xl font-bold text-white mb-1">Good</div>
            <div className="text-green-400 text-sm">AQI: 42</div>
          </div>

          <div className="card-dark rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Thermometer className="w-5 h-5 text-orange-400" />
                <span className="text-slate-300 text-sm">Temperature</span>
              </div>
              <Activity className="w-4 h-4 text-blue-400" />
            </div>
            <div className="text-2xl font-bold text-white mb-1">22.5°C</div>
            <div className="text-blue-400 text-sm">Optimal range</div>
          </div>
        </div>

        {/* Charts Row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Energy Consumption */}
          <div className="card-dark rounded-2xl p-6">
            <h3 className="text-white font-semibold text-lg mb-4">Energy Consumption (24h)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={energyData}>
                  <defs>
                    <linearGradient id="energyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#fbbf24" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Area
                    type="monotone"
                    dataKey="consumption"
                    stroke="#fbbf24"
                    strokeWidth={2}
                    fill="url(#energyGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Water Usage */}
          <div className="card-dark rounded-2xl p-6">
            <h3 className="text-white font-semibold text-lg mb-4">Water Usage (Weekly)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={waterUsageData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="day" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="usage" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Charts Row 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Air Quality Trends */}
          <div className="card-dark rounded-2xl p-6">
            <h3 className="text-white font-semibold text-lg mb-4">Air Quality Trends</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={airQualityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line type="monotone" dataKey="PM25" stroke="#ef4444" strokeWidth={2} name="PM2.5" />
                  <Line type="monotone" dataKey="PM10" stroke="#f97316" strokeWidth={2} name="PM10" />
                  <Line type="monotone" dataKey="CO2" stroke="#06b6d4" strokeWidth={2} name="CO₂" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Device Usage Distribution */}
          <div className="card-dark rounded-2xl p-6">
            <h3 className="text-white font-semibold text-lg mb-4">Device Usage Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={deviceUsageData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                    labelStyle={{ fill: "#fff", fontSize: "12px" }}
                  >
                    {deviceUsageData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Temperature & Humidity */}
        <div className="card-dark rounded-2xl p-6">
          <h3 className="text-white font-semibold text-lg mb-4">Temperature & Humidity (24h)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={temperatureData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis yAxisId="temp" orientation="left" stroke="#94a3b8" />
                <YAxis yAxisId="humidity" orientation="right" stroke="#94a3b8" />
                <Tooltip contentStyle={tooltipStyle} />
                <Line
                  yAxisId="temp"
                  type="monotone"
                  dataKey="temp"
                  stroke="#f97316"
                  strokeWidth={2}
                  name="Temperature (°C)"
                />
                <Line
                  yAxisId="humidity"
                  type="monotone"
                  dataKey="humidity"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  name="Humidity (%)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-slate-400 text-sm">
        KonnectSens Smart Home Hub • Analytics Dashboard
      </footer>
    </div>
  )
}

export default Analytics
