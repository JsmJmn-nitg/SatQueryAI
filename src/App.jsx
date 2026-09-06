import React, { useState, useRef } from "react";
import {
  Sparkles,
  Send,
  Layers,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Image as ImageIcon,
  Radar,
  Sun,
  Moon,
  UploadCloud,
  X,
  MessageSquare,
  History,
  Home,
  ChevronDown,
  Terminal,
  Download,
  Bot,
  BarChart3,
  Loader2
} from "lucide-react";

export default function SatQueryApp() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState("Single Image");
  const [query, setQuery] = useState("Analyze this imagery and describe all visible features, hazards, and land cover.");
  const [loading, setLoading] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showTraceModal, setShowTraceModal] = useState(false);
  const [responseTab, setResponseTab] = useState("executive");

  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [file1Name, setFile1Name] = useState("No image selected");
  const [file2Name, setFile2Name] = useState(null);

  const [analysisResult, setAnalysisResult] = useState({
    title: "SatQuery AI - Standby",
    executiveSummary: "Upload any satellite GeoTIFF or standard image (wildfire, flood, city, farmland) and click Send. The AI will inspect the pixels and generate a custom report.",
    confidenceScore: "0.92",
    previewUrl: "https://images.unsplash.com/photo-1524813686514-a57563d77d66?auto=format&fit=crop&w=1200&q=80",
    classDistribution: [
      { name: "Awaiting Image", percentage: 100, color: "#6366F1" }
    ],
    spectralMetrics: {
      "Status": "Ready for Analysis",
      "Format": "GeoTIFF / TIFF / PNG / JPG",
      "VLM Engine": "Qwen2.5-VL / BLIP Multimodal"
    },
    features: [],
    executionTrace: {}
  });

  const fileInputRef1 = useRef(null);
  const fileInputRef2 = useRef(null);

  const handleFileUpload = (e, target) => {
    const file = e.target.files[0];
    if (!file) return;

    if (target === 1) {
      setImage1(file);
      setFile1Name(file.name);
      // If it's a regular PNG/JPG, we can preview immediately; for TIFF, we wait for backend normalization
      if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
        setAnalysisResult((prev) => ({ ...prev, previewUrl: URL.createObjectURL(file) }));
      }
    } else {
      setImage2(file);
      setFile2Name(file.name);
    }
  };

  const executeAnalysis = async () => {
    if (!image1) {
      alert("Please select a GeoTIFF or image first!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("mode", activeTab);
    formData.append("query", query);
    formData.append("image1", image1);
    if (image2) formData.append("image2", image2);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Backend error: " + res.statusText);
      const data = await res.json();
      setAnalysisResult({
        title: data.title,
        executiveSummary: data.executive_summary,
        confidenceScore: data.confidence_score,
        previewUrl: data.preview_url,
        features: data.features,
        classDistribution: data.class_distribution,
        spectralMetrics: data.spectral_metrics,
        executionTrace: data.execution_summary
      });
    } catch (err) {
      alert("Error processing image: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex ${darkMode ? "bg-[#090D1A] text-slate-100" : "bg-[#F8FAFC] text-slate-800"}`}>
      {/* Sidebar */}
      <aside className={`w-64 border-r flex flex-col justify-between p-4 ${
        darkMode ? "bg-[#0B1021] border-[#1A233D]" : "bg-white border-slate-200"
      }`}>
        <div>
          <div className="flex items-center gap-3 px-2 py-3 mb-6">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-pink-500 flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-base tracking-tight leading-none">SatQuery AI</h1>
              <span className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-500"}`}>Real Vision Assistant</span>
            </div>
          </div>

          <nav className="space-y-1.5">
            <button className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium ${
              darkMode ? "bg-[#18213F] text-indigo-300" : "bg-indigo-50 text-indigo-600"
            }`}>
              <Home className="w-4 h-4" /> Home
            </button>
            <button
              onClick={() => setShowTraceModal(true)}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium ${
                darkMode ? "text-slate-400 hover:bg-[#151D37]" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <Terminal className="w-4 h-4 text-emerald-400" /> Execution Trace
            </button>
          </nav>
        </div>

        <button
          onClick={() => setDarkMode(!darkMode)}
          className={`w-full flex items-center justify-between p-2.5 rounded-xl border text-xs font-medium ${
            darkMode ? "bg-[#0F162E] border-[#1E294B] text-slate-300" : "bg-white border-slate-200 text-slate-700"
          }`}
        >
          <div className="flex items-center gap-2">
            {darkMode ? <Moon className="w-4 h-4 text-indigo-400" /> : <Sun className="w-4 h-4 text-amber-500" />}
            <span>{darkMode ? "Dark Mode" : "Light Mode"}</span>
          </div>
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        </button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        <div className="p-8 max-w-7xl w-full mx-auto space-y-6">
          {/* Query Box */}
          <div className={`p-6 rounded-2xl border ${
            darkMode ? "bg-[#0D1224] border-[#1C2648]" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              SatQuery Remote Sensing AI <span className="text-sm font-normal text-slate-400">• Multi-format GeoTIFF & Optical VLM</span>
            </h2>

            <div className={`flex items-center rounded-2xl border p-1.5 mb-4 ${
              darkMode ? "bg-[#090D1C] border-[#222E54]" : "bg-slate-50 border-slate-200"
            }`}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && executeAnalysis()}
                placeholder="Ask anything about the uploaded satellite image..."
                className={`w-full bg-transparent px-4 py-2.5 text-sm outline-none ${darkMode ? "text-white" : "text-slate-900"}`}
              />
              <button
                onClick={executeAnalysis}
                disabled={loading}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2.5 rounded-xl shadow font-semibold text-xs flex items-center gap-2"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                {loading ? "Processing..." : "Analyze"}
              </button>
            </div>

            {/* Mode selection */}
            <div className="flex flex-wrap gap-3 mb-4">
              {["Single Image", "Optical + SAR", "Change Detection", "Autofetch"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-xl text-xs font-medium border ${
                    activeTab === tab
                      ? darkMode ? "bg-[#1C2448] border-indigo-500 text-white" : "bg-indigo-50 border-indigo-400 text-indigo-700"
                      : darkMode ? "bg-[#0F152C] border-[#1C2648] text-slate-400" : "bg-white border-slate-200 text-slate-600"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Upload Buttons */}
            <div className="flex gap-4 items-center pt-4 border-t border-inherit">
              <div
                onClick={() => fileInputRef1.current.click()}
                className={`px-5 py-4 rounded-xl border-2 border-dashed flex items-center gap-3 cursor-pointer ${
                  darkMode ? "border-[#222E54] hover:border-indigo-500 bg-[#090D1C]" : "border-slate-300 hover:border-indigo-400 bg-slate-50"
                }`}
              >
                <input ref={fileInputRef1} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, 1)} className="hidden" />
                <UploadCloud className="w-5 h-5 text-indigo-400" />
                <div>
                  <p className="text-xs font-semibold">{file1Name}</p>
                  <p className="text-[10px] text-slate-500">Click to upload Primary GeoTIFF / TIFF / PNG</p>
                </div>
              </div>

              {(activeTab === "Optical + SAR" || activeTab === "Change Detection") && (
                <div
                  onClick={() => fileInputRef2.current.click()}
                  className={`px-5 py-4 rounded-xl border-2 border-dashed flex items-center gap-3 cursor-pointer ${
                    darkMode ? "border-[#222E54] hover:border-indigo-500 bg-[#090D1C]" : "border-slate-300 hover:border-indigo-400 bg-slate-50"
                  }`}
                >
                  <input ref={fileInputRef2} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, 2)} className="hidden" />
                  <UploadCloud className="w-5 h-5 text-purple-400" />
                  <div>
                    <p className="text-xs font-semibold">{file2Name || "Select Secondary Image"}</p>
                    <p className="text-[10px] text-slate-500">Pair Image (SAR or T2)</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* AI Response Card */}
          <div className={`p-6 rounded-2xl border ${darkMode ? "bg-[#0B1021] border-[#1A233D]" : "bg-white border-slate-200"}`}>
            <div className="flex justify-between items-center mb-4 pb-3 border-b border-inherit">
              <span className="text-xs font-semibold text-indigo-400 flex items-center gap-1.5">
                <Bot className="w-4 h-4" /> Vision-Language Model Finding
              </span>
              <span className="text-xs font-mono text-emerald-400">
                Confidence: {analysisResult.confidenceScore}
              </span>
            </div>

            <div className="mb-5">
              <h3 className="text-base font-bold mb-1">{analysisResult.title}</h3>
              <p className="text-sm leading-relaxed text-slate-300">{analysisResult.executiveSummary}</p>
            </div>

            {/* Graphs / Metric Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
              {Object.entries(analysisResult.spectralMetrics).map(([k, v], idx) => (
                <div key={idx} className={`p-3 rounded-xl border ${darkMode ? "bg-[#070B18] border-[#182242]" : "bg-slate-50 border-slate-200"}`}>
                  <span className="text-[10px] text-slate-400 block uppercase font-mono">{k}</span>
                  <span className="text-xs font-bold text-indigo-400">{v}</span>
                </div>
              ))}
            </div>

            {/* Visualizer & Vector Polygons */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              <div className="lg:col-span-8 rounded-2xl overflow-hidden border border-inherit relative bg-black">
                <div
                  className="relative w-full h-[400px] overflow-hidden"
                  style={{ transform: `scale(${zoomLevel})`, transformOrigin: "center center" }}
                >
                  <img src={analysisResult.previewUrl} alt="Satellite Output" className="w-full h-full object-cover" />
                  {showOverlays && (
                    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1024 1024" preserveAspectRatio="none">
                      {analysisResult.features.map((f) => (
                        <polygon key={f.id} points={f.points} fill={`${f.color}40`} stroke={f.color} strokeWidth="3.5" />
                      ))}
                    </svg>
                  )}
                </div>
              </div>

              {/* Dynamic Legend */}
              <div className="lg:col-span-4 space-y-3">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <BarChart3 className="w-3.5 h-3.5" /> Detected Class Distribution
                </h4>
                {analysisResult.classDistribution.map((item, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="flex items-center gap-1.5 font-medium">
                        <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: item.color }} />
                        {item.name}
                      </span>
                      <span className="font-mono text-slate-400">{item.percentage}%</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${item.percentage}%`, backgroundColor: item.color }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Trace Modal */}
      {showTraceModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50">
          <div className="bg-[#0B1021] border border-[#1F2B48] p-6 rounded-2xl max-w-xl w-full">
            <h3 className="font-bold text-sm mb-3 text-indigo-400">Auditable Execution Summary</h3>
            <pre className="bg-[#070A14] text-emerald-400 p-4 rounded-xl text-xs overflow-auto max-h-96">
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
