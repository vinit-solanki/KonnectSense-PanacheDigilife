"use client"

import { useState } from "react"
import { Link } from "react-router-dom"
import {
  Wifi,
  Bell,
  Settings,
  User,
  Wind,
  Thermometer,
  Droplets,
  Shield,
  Camera,
  Lock,
  Users,
  Home,
  Zap,
  BarChart3,
  Lightbulb,
  ShieldCheck,
  Droplet,
} from "lucide-react"

const Dashboard = () => {
  const [devices, setDevices] = useState([
    { id: 1, name: "Living Room Lights", location: "Living Room", icon: Lightbulb, active: true },
    { id: 2, name: "Security System", location: "Main Door", icon: ShieldCheck, active: true },
    { id: 3, name: "Water Monitor", location: "Kitchen", icon: Droplet, active: true },
    { id: 4, name: "Smart Outlet", location: "Bedroom", icon: Zap, active: false },
  ])

  const toggleDevice = (id) => {
    setDevices(devices.map((device) => (device.id === id ? { ...device, active: !device.active } : device)))
  }

  const activeDevices = devices.filter((d) => d.active).length

  return (
    <div className="min-h-screen bg-dashboard">
      {/* Header */}
      <header className="px-6 py-4 bg-slate-900/50 backdrop-blur-sm border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-cyan-500 rounded-xl flex items-center justify-center">
              <span className="text-white font-bold text-lg">K</span>
            </div>
            <div>
              <h1 className="text-white font-semibold text-lg">KonnectSens</h1>
              <p className="text-slate-400 text-sm">Smart Home Hub</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-slate-300">
              <Wifi className="w-4 h-4" />
              <span className="text-sm">Connected</span>
            </div>
            <Bell className="w-5 h-5 text-slate-400 cursor-pointer hover:text-white" />
            <Settings className="w-5 h-5 text-slate-400 cursor-pointer hover:text-white" />
            <User className="w-5 h-5 text-slate-400 cursor-pointer hover:text-white" />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6 space-y-6">
        {/* Top Row Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Air Quality Index */}
          <div className="card-dark rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Wind className="w-5 h-5 text-cyan-400" />
                <span className="text-white font-medium">Air Quality Index</span>
              </div>
              <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm">Good</span>
            </div>

            <div className="text-center mb-6">
              <div className="text-4xl font-bold text-white mb-2">42</div>
              <div className="text-slate-400 text-sm">Overall AQI Score</div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 text-sm">PM2.5</span>
                <span className="text-cyan-400 text-sm">12 μg/m³</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: "24%" }}></div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-300 text-sm">PM10</span>
                <span className="text-cyan-400 text-sm">18 μg/m³</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: "18%" }}></div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-300 text-sm">CO₂</span>
                <span className="text-cyan-400 text-sm">450 ppm</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: "45%" }}></div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-300 text-sm">VOCs</span>
                <span className="text-cyan-400 text-sm">0.2 mg/m³</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: "10%" }}></div>
              </div>
            </div>

            <div className="flex items-center space-x-2 mt-4 text-slate-400 text-xs">
              <div className="w-4 h-4 border border-slate-400 rounded flex items-center justify-center">
                <div className="w-1 h-1 bg-slate-400 rounded-full"></div>
              </div>
              <span>Last updated: 2 minutes ago</span>
            </div>
          </div>

          {/* Environment */}
          <div className="card-dark rounded-2xl p-6">
            <div className="flex items-center space-x-2 mb-6">
              <Thermometer className="w-5 h-5 text-orange-400" />
              <span className="text-white font-medium">Environment</span>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="text-center">
                <Thermometer className="w-8 h-8 text-orange-400 mx-auto mb-3" />
                <div className="text-3xl font-bold text-white mb-1">22.5°C</div>
                <div className="text-slate-400 text-sm">Temperature</div>
              </div>

              <div className="text-center">
                <Droplets className="w-8 h-8 text-blue-400 mx-auto mb-3" />
                <div className="text-3xl font-bold text-white mb-1">45%</div>
                <div className="text-slate-400 text-sm">Humidity</div>
              </div>
            </div>

            <div className="flex items-center justify-center space-x-2 mt-6 text-slate-400 text-sm">
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <span>Living Room</span>
            </div>
          </div>

          {/* Security */}
          <div className="card-dark rounded-2xl p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-2">
                <Shield className="w-5 h-5 text-cyan-400" />
                <span className="text-white font-medium">Security</span>
              </div>
              <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm">Armed</span>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="text-center">
                <Camera className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
                <div className="text-2xl font-bold text-white">4</div>
                <div className="text-slate-400 text-xs">Cameras</div>
              </div>

              <div className="text-center">
                <Lock className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
                <div className="text-2xl font-bold text-white">8</div>
                <div className="text-slate-400 text-xs">Sensors</div>
              </div>

              <div className="text-center">
                <Users className="w-6 h-6 text-green-400 mx-auto mb-2" />
                <div className="text-lg font-bold text-green-400">Safe</div>
                <div className="text-slate-400 text-xs">Status</div>
              </div>
            </div>

            <div className="text-slate-400 text-sm text-center">Last event: 2 hours ago</div>
          </div>
        </div>

        {/* Bottom Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Smart Devices */}
          <div className="lg:col-span-2 card-dark rounded-2xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-white font-semibold text-xl">Smart Devices</h2>
              <span className="text-slate-400 text-sm">{activeDevices}/4 Active</span>
            </div>

            <div className="space-y-4">
              {devices.map((device) => {
                const Icon = device.icon
                return (
                  <div key={device.id} className="flex items-center justify-between p-4 rounded-xl bg-slate-800/50">
                    <div className="flex items-center space-x-4">
                      <div className="w-10 h-10 bg-slate-700 rounded-lg flex items-center justify-center">
                        <Icon className="w-5 h-5 text-cyan-400" />
                      </div>
                      <div>
                        <div className="text-white font-medium">{device.name}</div>
                        <div className="text-slate-400 text-sm">{device.location}</div>
                      </div>
                    </div>

                    <div
                      className={`toggle-switch ${device.active ? "active" : ""}`}
                      onClick={() => toggleDevice(device.id)}
                    ></div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card-dark rounded-2xl p-6">
            <h2 className="text-white font-semibold text-xl mb-6">Quick Actions</h2>

            <div className="grid grid-cols-3 gap-4">
              <button className="flex flex-col items-center space-y-2 p-4 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 transition-colors">
                <Home className="w-6 h-6 text-cyan-400" />
                <span className="text-slate-300 text-xs">All Devices</span>
              </button>

              <button className="flex flex-col items-center space-y-2 p-4 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 transition-colors">
                <Zap className="w-6 h-6 text-yellow-400" />
                <span className="text-slate-300 text-xs">Energy</span>
              </button>

              <button className="flex flex-col items-center space-y-2 p-4 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 transition-colors">
                <Shield className="w-6 h-6 text-red-400" />
                <span className="text-slate-300 text-xs">Security</span>
              </button>

              <button className="flex flex-col items-center space-y-2 p-4 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 transition-colors">
                <Users className="w-6 h-6 text-green-400" />
                <span className="text-slate-300 text-xs">Family</span>
              </button>

              <Link
                to="/analytics"
                className="flex flex-col items-center space-y-2 p-4 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 transition-colors"
              >
                <BarChart3 className="w-6 h-6 text-blue-400" />
                <span className="text-slate-300 text-xs">Analytics</span>
              </Link>

              <button className="flex flex-col items-center space-y-2 p-4 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 transition-colors">
                <Settings className="w-6 h-6 text-slate-400" />
                <span className="text-slate-300 text-xs">Settings</span>
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-slate-400 text-sm">
        KonnectSens Smart Home Hub • All systems operational
      </footer>
    </div>
  )
}

export default Dashboard
