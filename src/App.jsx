import React, { useState, useRef } from "react";
import {
  Sparkles,
  Send,
  Layers,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Sun,
  Moon,
  UploadCloud,
  Home,
  ChevronDown,
  Terminal,
  Bot,
  BarChart3,
  Loader2,
  Waves,
  Building2,
  AlertTriangle,
  FileText
} from "lucide-react";

export default function SatQueryApp() {
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState("Single Image");
  const [query, setQuery] = useState("Identify the top 4 critical features and hazards. How many rivers are visible, and what percentage of the image is covered by urban settlement?");
  const [loading, setLoading] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showTraceModal, setShowTraceModal] = useState(false);
  const [hoveredIdx, setHoveredIdx] = useState(null);

  const [image1, setImage1] = useState(null);
  const [file1Name, setFile1Name] = useState("No image selected");

  const [analysisResult, setAnalysisResult] = useState({
    title: "SatQuery AI - Standby",
    directQueryAnswers: {
      hydrology_and_waterways: "Upload an image to inspect hydrological networks and surface water.",
      urban_settlement_coverage: "Upload an image to evaluate urban footprint and infrastructure density.",
      hazards_and_vulnerabilities: "Upload an image to identify environmental, thermal, or coastal hazards."
    },
    comprehensiveAssessment: "SatQuery AI automatically classifies the scene domain (Wildfire, Coastal, Flood, Urban) and computes grounded spatial coordinates.",
    confidenceScore: "0.95",
    previewUrl: "https://images.unsplash.com/photo-1524813686514-a57563d77d66?auto=format&fit=crop&w=1200&q=80",
    classDistribution: [
      { name: "Awaiting Image", percentage: 100, color: "#6366F1", description: "Upload any satellite TIFF, GeoTIFF, or photo." }
    ],
    spectralMetrics: {
      "System Status": "Operational",
      "Model": "Qwen2-VL-2B (Adaptive Grounding)",
      "Resolution": "Sub-pixel Adaptive"
    },
    features: [],
    executionTrace: {}
  });

  const fileInputRef1 = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImage1(file);
    setFile1Name(file.name);
    if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
      setAnalysisResult((prev) => ({ ...prev, previewUrl: URL.createObjectURL(file) }));
    }
  };

  const executeAnalysis = async () => {
    if (!image1) {
      alert("Please upload an image first!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("mode", activeTab);
    formData.append("query", query);
    formData.append("image1", image1);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Server error: " + res.statusText);
      const data = await res.json();
      setAnalysisResult({
        title: data.title,
        directQueryAnswers: data.direct_query_answers || {},
        comprehensiveAssessment: data.comprehensive_assessment,
        confidenceScore: data.confidence_score,
        previewUrl: data.preview_url,
        features: data.features,
        classDistribution: data.class_distribution,
        spectralMetrics: data.spectral_metrics,
        executionTrace: data.execution_summary
      });
    } catch (err) {
      alert("Analysis error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex ${darkMode ? "bg-[#090D1A] text-slate-100" : "bg-[#F8FAFC] text-slate-900"}`}>
      {/* Sidebar */}
      <aside className={`w-64 border-r flex flex-col justify-between p-4 ${
        darkMode ? "bg-[#0B1021] border-[#1A233D]" : "bg-white border-slate-200 shadow-sm"
      }`}>
        <div>
          <div className="flex items-center gap-3 px-2 py-3 mb-6">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-pink-500 flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-base tracking-tight leading-none text-slate-900 dark:text-white">SatQuery AI</h1>
              <span className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-600 font-semibold"}`}>Earth Observation AI</span>
            </div>
          </div>

          <nav className="space-y-1.5">
            <button className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-bold ${
              darkMode ? "bg-[#18213F] text-indigo-300" : "bg-indigo-50 text-indigo-700"
            }`}>
              <Home className="w-4 h-4" /> Home
            </button>
            <button
              onClick={() => setShowTraceModal(true)}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold ${
                darkMode ? "text-slate-400 hover:bg-[#151D37]" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              <Terminal className="w-4 h-4 text-emerald-600 dark:text-emerald-400" /> Execution Trace
            </button>
          </nav>
        </div>

        <button
          onClick={() => setDarkMode(!darkMode)}
          className={`w-full flex items-center justify-between p-2.5 rounded-xl border text-xs font-bold ${
            darkMode ? "bg-[#0F162E] border-[#1E294B] text-slate-300" : "bg-slate-50 border-slate-300 text-slate-800"
          }`}
        >
          <div className="flex items-center gap-2">
            {darkMode ? <Moon className="w-4 h-4 text-indigo-400" /> : <Sun className="w-4 h-4 text-amber-500" />}
            <span>{darkMode ? "Dark Mode" : "Light Mode"}</span>
          </div>
          <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
        </button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        <div className="p-8 max-w-7xl w-full mx-auto space-y-6">
          {/* Query & Upload Box */}
          <div className={`p-6 rounded-2xl border ${
            darkMode ? "bg-[#0D1224] border-[#1C2648]" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2 text-slate-900 dark:text-white">
              Universal Satellite Intelligence <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-400">• Multi-Scene Reasoning</span>
            </h2>

            <div className={`flex items-center rounded-2xl border p-1.5 mb-4 ${
              darkMode ? "bg-[#090D1C] border-[#222E54]" : "bg-slate-50 border-slate-300"
            }`}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && executeAnalysis()}
                placeholder="Ask specific questions about fires, floods, rivers, urban %, hazards..."
                className={`w-full bg-transparent px-4 py-2.5 text-sm outline-none font-medium ${darkMode ? "text-white" : "text-slate-900"}`}
              />
              <button
                onClick={executeAnalysis}
                disabled={loading}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2.5 rounded-xl shadow font-bold text-xs flex items-center gap-2 shrink-0"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>

            {/* Mode selection */}
            <div className="flex flex-wrap gap-2 mb-4">
              {["Single Image", "Optical + SAR", "Change Detection", "Autofetch"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold border ${
                    activeTab === tab
                      ? darkMode ? "bg-[#1C2448] border-indigo-500 text-white" : "bg-indigo-50 border-indigo-600 text-indigo-700 shadow-sm"
                      : darkMode ? "bg-[#0F152C] border-[#1C2648] text-slate-400" : "bg-white border-slate-300 text-slate-700"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Upload Area */}
            <div className="pt-4 border-t border-inherit">
              <div
                onClick={() => fileInputRef1.current.click()}
                className={`px-5 py-4 rounded-xl border-2 border-dashed flex items-center gap-3 cursor-pointer ${
                  darkMode ? "border-[#222E54] hover:border-indigo-500 bg-[#090D1C]" : "border-slate-300 hover:border-indigo-600 bg-slate-50"
                }`}
              >
                <input ref={fileInputRef1} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={handleFileUpload} className="hidden" />
                <UploadCloud className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                <div>
                  <p className="text-xs font-bold text-slate-900 dark:text-slate-200">{file1Name}</p>
                  <p className="text-[10px] text-slate-600 dark:text-slate-400 font-medium">Upload any TIFF, GeoTIFF, PNG or JPEG satellite image</p>
                </div>
              </div>
            </div>
          </div>

          {/* AI Response Card */}
          <div className={`p-6 rounded-2xl border ${darkMode ? "bg-[#0B1021] border-[#1A233D]" : "bg-white border-slate-200 shadow-sm"}`}>
            <div className="flex justify-between items-center mb-4 pb-3 border-b border-inherit">
              <span className="text-xs font-bold text-indigo-700 dark:text-indigo-400 flex items-center gap-1.5">
                <Bot className="w-4 h-4" /> Comprehensive Geospatial Assessment
              </span>
              <span className="text-xs font-mono text-emerald-700 dark:text-emerald-400 font-bold">
                Confidence: {analysisResult.confidenceScore}
              </span>
            </div>

            <h3 className="text-lg font-extrabold mb-4 text-slate-900 dark:text-white">{analysisResult.title}</h3>

            {/* DIRECT QUERY Q&A SECTION */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
              {/* Hydrology */}
              <div className={`p-3.5 rounded-xl border ${darkMode ? "bg-[#0E152E] border-sky-500/20" : "bg-sky-50 border-sky-300 shadow-sm"}`}>
                <div className="flex items-center gap-2 mb-1.5">
                  <Waves className="w-4 h-4 text-sky-700 dark:text-sky-400" />
                  <span className="text-xs font-extrabold text-sky-950 dark:text-sky-300">Hydrological & River Analysis</span>
                </div>
                <p className="text-xs text-slate-800 dark:text-slate-300 leading-relaxed font-medium">
                  {analysisResult.directQueryAnswers.hydrology_and_waterways}
                </p>
              </div>

              {/* Urban */}
              <div className={`p-3.5 rounded-xl border ${darkMode ? "bg-[#0E152E] border-rose-500/20" : "bg-rose-50 border-rose-300 shadow-sm"}`}>
                <div className="flex items-center gap-2 mb-1.5">
                  <Building2 className="w-4 h-4 text-rose-700 dark:text-rose-400" />
                  <span className="text-xs font-extrabold text-rose-950 dark:text-rose-300">Urban Settlement Coverage</span>
                </div>
                <p className="text-xs text-slate-800 dark:text-slate-300 leading-relaxed font-medium">
                  {analysisResult.directQueryAnswers.urban_settlement_coverage}
                </p>
              </div>

              {/* Hazards */}
              <div className={`p-3.5 rounded-xl border ${darkMode ? "bg-[#0E152E] border-amber-500/20" : "bg-amber-50 border-amber-300 shadow-sm"}`}>
                <div className="flex items-center gap-2 mb-1.5">
                  <AlertTriangle className="w-4 h-4 text-amber-700 dark:text-amber-400" />
                  <span className="text-xs font-extrabold text-amber-950 dark:text-amber-300">Hazards & Vulnerabilities</span>
                </div>
                <p className="text-xs text-slate-800 dark:text-slate-300 leading-relaxed font-medium">
                  {analysisResult.directQueryAnswers.hazards_and_vulnerabilities}
                </p>
              </div>
            </div>

            {/* FULL MULTI-PARAGRAPH ASSESSMENT */}
            <div className={`p-4 rounded-xl border mb-6 ${
              darkMode ? "bg-[#070A16] border-[#1A233D]" : "bg-slate-50 border-slate-300 shadow-sm"
            }`}>
              <div className="flex items-center gap-2 mb-2 text-xs font-extrabold text-slate-700 dark:text-slate-400 uppercase tracking-wider">
                <FileText className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> Technical Intelligence Report
              </div>
              <p className="text-xs leading-relaxed text-slate-900 dark:text-slate-200 whitespace-pre-line font-medium">
                {analysisResult.comprehensiveAssessment}
              </p>
            </div>

            {/* Spectral Indices */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
              {Object.entries(analysisResult.spectralMetrics).map(([k, v], idx) => (
                <div key={idx} className={`p-3 rounded-xl border ${darkMode ? "bg-[#070B18] border-[#182242]" : "bg-white border-slate-300 shadow-sm"}`}>
                  <span className="text-[10px] text-slate-600 dark:text-slate-400 block uppercase font-mono font-bold">{k}</span>
                  <span className="text-xs font-extrabold text-indigo-700 dark:text-indigo-400">{v}</span>
                </div>
              ))}
            </div>

            {/* Interactive Grid: Visualizer + 4 Classes */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Image Visualizer with Vector Overlays */}
              <div className="lg:col-span-8 rounded-2xl overflow-hidden border border-inherit relative bg-black">
                <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-slate-900/80 backdrop-blur-md p-1.5 rounded-xl border border-white/10 shadow-lg">
                  <button onClick={() => setZoomLevel((z) => Math.min(z + 0.2, 2.2))} className="p-1.5 text-slate-300 hover:text-white" title="Zoom In">
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => setZoomLevel((z) => Math.max(z - 0.2, 0.8))} className="p-1.5 text-slate-300 hover:text-white" title="Zoom Out">
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => setShowOverlays(!showOverlays)} className={`p-1.5 rounded-lg ${showOverlays ? "text-indigo-400 bg-indigo-500/20" : "text-slate-300"}`} title="Toggle Grounding">
                    <Layers className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div
                  className="relative w-full h-[440px] overflow-hidden"
                  style={{ transform: `scale(${zoomLevel})`, transformOrigin: "center center" }}
                >
                  <img src={analysisResult.previewUrl} alt="Satellite Scene" className="w-full h-full object-cover select-none" />

                  {showOverlays && (
                    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1024 1024" preserveAspectRatio="none">
                      {analysisResult.features.map((f, idx) => {
                        const isHovered = hoveredIdx === idx;
                        return (
                          <g key={f.id}>
                            <polygon
                              points={f.points}
                              fill={isHovered ? `${f.color}66` : `${f.color}33`}
                              stroke={f.color}
                              strokeWidth={isHovered ? "5" : "3.5"}
                              strokeDasharray={idx === 1 ? "6,4" : "none"}
                            />
                            <circle cx={f.center[0]} cy={f.center[1]} r="16" fill={f.color} stroke="#FFFFFF" strokeWidth="2" />
                            <text
                              x={f.center[0]}
                              y={f.center[1] + 5}
                              textAnchor="middle"
                              fill="#FFFFFF"
                              fontSize="14"
                              fontWeight="bold"
                            >
                              {idx + 1}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                </div>
              </div>

              {/* 4 DYNAMIC METRICS LIST */}
              <div className="lg:col-span-4 space-y-3.5">
                <div className="flex items-center justify-between pb-1 border-b border-inherit">
                  <h4 className="text-xs font-extrabold uppercase tracking-wider text-slate-800 dark:text-slate-400 flex items-center gap-1.5">
                    <BarChart3 className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> Detected Class Distribution
                  </h4>
                  <span className="text-[10px] text-slate-600 dark:text-slate-500 font-mono font-bold">4 Discrete Metrics</span>
                </div>

                {analysisResult.classDistribution.map((item, idx) => {
                  const isHovered = hoveredIdx === idx;
                  return (
                    <div
                      key={idx}
                      onMouseEnter={() => setHoveredIdx(idx)}
                      onMouseLeave={() => setHoveredIdx(null)}
                      className={`p-2.5 rounded-xl border transition-all cursor-pointer ${
                        isHovered
                          ? "border-indigo-600 bg-indigo-50/50 dark:bg-indigo-500/10 shadow-md"
                          : darkMode ? "border-[#1A233D] bg-[#070A16]" : "border-slate-300 bg-white shadow-sm"
                      }`}
                    >
                      <div className="flex justify-between items-center text-xs mb-1.5">
                        <span className="flex items-center gap-2 font-bold text-slate-900 dark:text-slate-200">
                          <span
                            className="w-4 h-4 rounded-md text-[10px] font-bold text-white flex items-center justify-center shrink-0"
                            style={{ backgroundColor: item.color }}
                          >
                            {idx + 1}
                          </span>
                          <span className="truncate max-w-[170px]" title={item.name}>
                            {item.name}
                          </span>
                        </span>
                        <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-300">
                          {item.percentage}%
                        </span>
                      </div>

                      <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden mb-1.5">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${item.percentage}%`, backgroundColor: item.color }}
                        />
                      </div>

                      {item.description && (
                        <p className="text-[10px] text-slate-700 dark:text-slate-400 leading-tight font-medium">
                          {item.description}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Trace Modal */}
      {showTraceModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-[#0B1021] border border-slate-200 dark:border-[#1F2B48] p-6 rounded-2xl max-w-xl w-full text-slate-900 dark:text-white">
            <h3 className="font-bold text-sm mb-3 text-indigo-600 dark:text-indigo-400">Auditable Grounding Trace</h3>
            <pre className="bg-slate-900 text-emerald-400 p-4 rounded-xl text-xs overflow-auto max-h-96">
              {JSON.stringify(analysisResult.executionTrace, null, 2)}
            </pre>
            <button onClick={() => setShowTraceModal(false)} className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold">
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
