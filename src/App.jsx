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
  HelpCircle,
  History,
  Home,
  ChevronDown,
  Terminal
} from "lucide-react";

export default function SatQueryApp() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState("Single Image");
  const [query, setQuery] = useState('What are the main land cover types in this image?');
  const [loading, setLoading] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showTraceModal, setShowTraceModal] = useState(false);

  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [fileDetails1, setFileDetails1] = useState({
    name: "optical_image.tif",
    size: "10.4 MB",
    dimensions: "1024×1024"
  });

  const [analysisResult, setAnalysisResult] = useState({
    title: "Coastal Land-Cover Overview",
    summary:
      "This image shows a coastal region with a mix of urban, agricultural, and natural land-cover types. Major objects include:",
    confidenceScore: "0.88",
    previewUrl:
      "https://images.unsplash.com/photo-1524813686514-a57563d77d66?auto=format&fit=crop&w=1200&q=80",
    features: [
      {
        id: "built-up",
        name: "Built-up area",
        desc: "Dense urban settlement along the coast and inland.",
        color: "#EF4444",
        points: "550,420 680,410 660,650 560,640"
      },
      {
        id: "water",
        name: "Water body",
        desc: "Sea/ocean on the left side and small inland water bodies.",
        color: "#0EA5E9",
        points: "20,50 180,60 160,850 10,850"
      },
      {
        id: "vegetation",
        name: "Vegetation",
        desc: "Green patches of dense vegetation and agricultural fields.",
        color: "#10B981",
        points: "220,70 360,60 340,300 230,310"
      },
      {
        id: "roads",
        name: "Roads",
        desc: "Major road network connecting urban areas.",
        color: "#F59E0B",
        points: "200,670 420,680 780,490 690,470 210,650"
      },
      {
        id: "bare-land",
        name: "Bare land",
        desc: "Some areas of exposed soil or sparse vegetation.",
        color: "#A855F7",
        points: "690,690 780,680 770,820 680,810"
      }
    ],
    executionTrace: {
      task: "single_image_grounded_vqa",
      inputs: { n_images: 1, mode: "Single Image", format: "GeoTIFF" },
      tools_used: [
        { name: "GeoChat-7B", params: { temperature: 0.2, top_p: 0.9 } },
        { name: "GroundedSegmentationTool", params: { iou_threshold: 0.45 } }
      ],
      metrics: { confidence_score: 0.88 },
      notes: ["Single GeoTIFF loaded; CRS verified EPSG:4326"]
    }
  });

  const fileInputRef1 = useRef(null);
  const fileInputRef2 = useRef(null);

  const handleFileUpload = (e, target) => {
    const file = e.target.files[0];
    if (!file) return;

    const details = {
      name: file.name,
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      dimensions: "1024×1024"
    };

    if (target === 1) {
      setImage1(file);
      setFileDetails1(details);
    } else {
      setImage2(file);
    }
  };

  const handleExecuteQuery = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append("mode", activeTab);
    formData.append("query", query);
    if (image1) formData.append("image1", image1);
    if (image2) formData.append("image2", image2);

    try {
      // Relative path: routes directly through the FastAPI server and tunnel
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) throw new Error("Backend processing error");

      const data = await response.json();
      setAnalysisResult({
        title: data.title,
        summary: data.summary,
        confidenceScore: data.confidence_score.toString(),
        previewUrl: data.preview_url,
        features: data.features.map((f, i) => ({
          id: f.id,
          name: f.name,
          desc: f.description,
          color: f.color,
          points:
            analysisResult.features[i]?.points || "300,300 400,300 400,400 300,400"
        })),
        executionTrace: data.execution_summary
      });
    } catch (err) {
      console.warn("Backend unavailable or in transition:", err);
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
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-base tracking-tight leading-none">SatQuery AI</h1>
              <span className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                Vision–Language Assistant
              </span>
            </div>
          </div>

          <button
            onClick={() => {
              setQuery("");
              setImage2(null);
            }}
            className="w-full mb-6 py-2.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-medium text-sm flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/30 hover:opacity-95 transition-all"
          >
            <span className="text-lg leading-none">+</span> New Query
          </button>

          <nav className="space-y-1.5">
            <button
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                darkMode ? "bg-[#18213F] text-indigo-300" : "bg-indigo-50 text-indigo-600"
              }`}
            >
              <Home className="w-4 h-4" /> Home
            </button>
            <button
              onClick={() => setShowTraceModal(true)}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                darkMode ? "text-slate-400 hover:bg-[#151D37] hover:text-slate-200" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <Terminal className="w-4 h-4" /> Execution Trace
            </button>
            <button
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                darkMode ? "text-slate-400 hover:bg-[#151D37] hover:text-slate-200" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <History className="w-4 h-4" /> History
            </button>
          </nav>
        </div>

        <div className="space-y-3 pt-4 border-t border-inherit">
          <div className={`p-3 rounded-xl border flex items-center gap-2.5 ${
            darkMode ? "bg-[#0F162E] border-[#1E294B]" : "bg-slate-50 border-slate-200"
          }`}>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <div>
              <p className="text-xs font-semibold">System Status</p>
              <p className={`text-[11px] ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                All systems operational
              </p>
            </div>
          </div>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`w-full flex items-center justify-between p-2.5 rounded-xl border text-xs font-medium ${
              darkMode ? "bg-[#0F162E] border-[#1E294B] text-slate-300 hover:border-slate-600" : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
            }`}
          >
            <div className="flex items-center gap-2">
              {darkMode ? <Moon className="w-4 h-4 text-indigo-400" /> : <Sun className="w-4 h-4 text-amber-500" />}
              <span>{darkMode ? "Dark Mode" : "Light Mode"}</span>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
      </aside>

      {/* Main Canvas */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        <header className={`h-16 border-b flex items-center justify-end px-8 gap-5 ${
          darkMode ? "border-[#1A233D] bg-[#090D1A]" : "border-slate-200 bg-white"
        }`}>
          <button className={`flex items-center gap-1.5 text-xs font-medium ${
            darkMode ? "text-slate-400 hover:text-slate-200" : "text-slate-600 hover:text-slate-900"
          }`}>
            <HelpCircle className="w-4 h-4" /> Help
          </button>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-lg border ${
              darkMode ? "border-[#1E294B] text-slate-300 hover:bg-[#141C38]" : "border-slate-200 text-slate-600 hover:bg-slate-100"
            }`}
          >
            {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          <div className="flex items-center gap-2 pl-3 border-l border-inherit">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 text-white font-semibold text-xs flex items-center justify-center">
              U
            </div>
            <span className="text-xs font-medium">User</span>
            <ChevronDown className="w-3 h-3 text-slate-400" />
          </div>
        </header>

        <div className="p-8 max-w-7xl w-full mx-auto space-y-6">
          <div className={`relative p-6 rounded-2xl border overflow-hidden ${
            darkMode ? "bg-gradient-to-b from-[#11172E] to-[#0D1224] border-[#1C2648]" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <div className="absolute right-6 top-1/2 -translate-y-1/2 pointer-events-none opacity-40 md:opacity-90">
              <div className="relative w-64 h-36 flex items-center justify-center">
                <div className="absolute w-52 h-20 border border-indigo-500/20 rounded-[100%] rotate-[-18deg]" />
                <div className="absolute w-64 h-24 border border-purple-500/30 rounded-[100%] rotate-[-22deg]" />
                <div className="w-14 h-14 rounded-full bg-gradient-to-br from-indigo-400 via-purple-600 to-slate-900 shadow-xl shadow-indigo-500/30" />
              </div>
            </div>

            <div className="relative z-10 mb-4">
              <h2 className="text-xl font-bold flex items-center gap-2">
                Good morning! <span className="text-2xl">👋</span>
                <span className={`text-sm font-normal ml-2 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                  Ask anything about your remote sensing imagery.
                </span>
              </h2>
            </div>

            <div className="relative z-10 max-w-3xl mb-5">
              <div className={`flex items-center rounded-2xl border p-1.5 focus-within:ring-2 focus-within:ring-indigo-500 transition-all ${
                darkMode ? "bg-[#090D1C] border-[#222E54]" : "bg-slate-50 border-slate-200"
              }`}>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleExecuteQuery()}
                  placeholder='Try: "What are the main land cover types in this image?"'
                  className={`w-full bg-transparent px-4 py-2.5 text-sm outline-none placeholder:text-slate-500 ${
                    darkMode ? "text-white" : "text-slate-900"
                  }`}
                />
                <button
                  onClick={handleExecuteQuery}
                  disabled={loading}
                  className="bg-indigo-600 hover:bg-indigo-500 text-white p-2.5 rounded-xl shadow-md transition-all flex items-center justify-center"
                >
                  <Send className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                </button>
              </div>
            </div>

            <div className="relative z-10 flex flex-wrap items-center gap-3">
              {[
                { id: "Single Image", icon: ImageIcon, label: "Single Image" },
                { id: "Optical + SAR", icon: Radar, label: "Optical + SAR" },
                { id: "Change Detection", icon: Layers, label: "Change Detection" },
                { id: "Autofetch", icon: Sparkles, label: "Autofetch" }
              ].map((tab) => {
                const Icon = tab.icon;
                const isSelected = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium border transition-all ${
                      isSelected
                        ? darkMode
                          ? "bg-[#1C2448] border-indigo-500 text-white shadow-md shadow-indigo-500/10"
                          : "bg-indigo-50 border-indigo-400 text-indigo-700"
                        : darkMode
                        ? "bg-[#0F152C] border-[#1C2648] text-slate-400 hover:text-slate-200"
                        : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    <Icon className={`w-3.5 h-3.5 ${isSelected ? "text-indigo-400" : ""}`} />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            <div className="mt-5 pt-4 border-t border-inherit">
              {activeTab === "Autofetch" ? (
                <div className={`p-4 rounded-xl border flex items-start gap-3 ${
                  darkMode ? "bg-[#090D1C]/80 border-[#1E294B]" : "bg-indigo-50/50 border-indigo-100"
                }`}>
                  <Sparkles className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-semibold text-indigo-400">Autofetch Mode Active</h4>
                    <p className={`text-xs mt-0.5 ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                      Describe your area of interest (e.g. <i>"Assess flood damage in Valencia after October 2024"</i>),
                      and SatQuery AI will automatically resolve the coordinates, fetch the required Sentinel-1/2
                      GeoTIFF bands, and execute the analysis.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-wrap gap-4 items-center">
                  <div
                    onClick={() => fileInputRef1.current.click()}
                    className={`relative w-48 h-28 rounded-xl border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-colors p-3 ${
                      darkMode ? "border-[#222E54] hover:border-indigo-500 bg-[#090D1C]" : "border-slate-300 hover:border-indigo-400 bg-slate-50"
                    }`}
                  >
                    <input
                      ref={fileInputRef1}
                      type="file"
                      accept=".tif,.tiff,.png,.jpg"
                      onChange={(e) => handleFileUpload(e, 1)}
                      className="hidden"
                    />
                    <img
                      src={analysisResult.previewUrl}
                      alt="Thumbnail"
                      className="absolute inset-0 w-full h-full object-cover rounded-xl opacity-20"
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setImage1(null);
                      }}
                      className="absolute top-1.5 right-1.5 p-1 rounded-full bg-black/50 hover:bg-black text-white"
                    >
                      <X className="w-3 h-3" />
                    </button>
                    <p className="text-xs font-semibold text-center z-10 truncate max-w-[140px]">
                      {fileDetails1.name}
                    </p>
                    <span className={`text-[10px] z-10 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                      {fileDetails1.size} • {fileDetails1.dimensions}
                    </span>
                  </div>

                  {(activeTab === "Optical + SAR" || activeTab === "Change Detection") && (
                    <div
                      onClick={() => fileInputRef2.current.click()}
                      className={`w-48 h-28 rounded-xl border-2 border-dashed flex flex-col items-center justify-center cursor-pointer p-3 transition-colors ${
                        image2
                          ? "border-emerald-500 bg-emerald-500/10"
                          : darkMode
                          ? "border-[#222E54] hover:border-indigo-500 bg-[#090D1C]"
                          : "border-slate-300 hover:border-indigo-400 bg-slate-50"
                      }`}
                    >
                      <input
                        ref={fileInputRef2}
                        type="file"
                        accept=".tif,.tiff,.png,.jpg"
                        onChange={(e) => handleFileUpload(e, 2)}
                        className="hidden"
                      />
                      <UploadCloud className="w-5 h-5 text-indigo-400 mb-1" />
                      <p className="text-xs font-semibold text-center">
                        {image2 ? image2.name : `Upload ${activeTab === "Optical + SAR" ? "SAR (TIFF)" : "T2 Image"}`}
                      </p>
                      <span className={`text-[10px] ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                        GeoTIFF / TIFF • Max 200MB
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Results Panel */}
          <div className={`p-6 rounded-2xl border ${
            darkMode ? "bg-[#0B1021] border-[#1A233D]" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <div className="flex items-center gap-2 mb-4">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <MessageSquare className="w-3.5 h-3.5" /> Answer
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              <div className="lg:col-span-5 space-y-5">
                <div>
                  <h3 className="text-lg font-bold">{analysisResult.title}</h3>
                  <p className={`text-xs mt-1.5 leading-relaxed ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                    {analysisResult.summary}
                  </p>
                </div>

                <div className="space-y-3.5">
                  {analysisResult.features.map((item) => (
                    <div key={item.id} className="flex items-start gap-3 text-xs">
                      <div
                        className="w-5 h-5 rounded-md mt-0.5 flex items-center justify-center shrink-0"
                        style={{ backgroundColor: `${item.color}25`, color: item.color }}
                      >
                        <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: item.color }} />
                      </div>
                      <div>
                        <span className="font-semibold">{item.name}</span>
                        <p className={darkMode ? "text-slate-400" : "text-slate-500"}>
                          {item.desc}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="pt-2 flex items-center gap-2.5">
                  <span className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                    Confidence Score
                  </span>
                  <span className="px-3 py-1 rounded-lg text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                    {analysisResult.confidenceScore}
                  </span>
                </div>
              </div>

              <div className="lg:col-span-7 flex flex-col md:flex-row gap-4">
                <div className={`relative flex-1 rounded-2xl overflow-hidden border ${
                  darkMode ? "border-[#1C2648] bg-black" : "border-slate-200 bg-slate-100"
                }`}>
                  <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-slate-900/80 backdrop-blur-md p-1.5 rounded-xl border border-white/10 shadow-lg">
                    <button
                      onClick={() => setZoomLevel((z) => Math.min(z + 0.2, 2.0))}
                      className="p-1.5 text-slate-300 hover:text-white hover:bg-white/10 rounded-lg transition"
                      title="Zoom In"
                    >
                      <ZoomIn className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setZoomLevel((z) => Math.max(z - 0.2, 0.8))}
                      className="p-1.5 text-slate-300 hover:text-white hover:bg-white/10 rounded-lg transition"
                      title="Zoom Out"
                    >
                      <ZoomOut className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setShowOverlays(!showOverlays)}
                      className={`p-1.5 rounded-lg transition ${
                        showOverlays ? "text-indigo-400 bg-indigo-500/20" : "text-slate-300 hover:bg-white/10"
                      }`}
                      title="Toggle Vector Layers"
                    >
                      <Layers className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setZoomLevel(1)}
                      className="p-1.5 text-slate-300 hover:text-white hover:bg-white/10 rounded-lg transition"
                      title="Reset View"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div
                    className="relative w-full h-[360px] overflow-hidden transition-transform duration-200"
                    style={{ transform: `scale(${zoomLevel})`, transformOrigin: "center center" }}
                  >
                    <img
                      src={analysisResult.previewUrl}
                      alt="Satellite Preview"
                      className="w-full h-full object-cover select-none"
                    />

                    {showOverlays && (
                      <svg
                        className="absolute inset-0 w-full h-full pointer-events-none"
                        viewBox="0 0 1024 1024"
                        preserveAspectRatio="none"
                      >
                        {analysisResult.features.map((feature) => (
                          <polygon
                            key={feature.id}
                            points={feature.points}
                            fill={`${feature.color}33`}
                            stroke={feature.color}
                            strokeWidth="3.5"
                            strokeDasharray={feature.id === "roads" ? "6,4" : "none"}
                          />
                        ))}
                      </svg>
                    )}
                  </div>
                </div>

                <div className={`p-4 rounded-xl border w-full md:w-44 shrink-0 ${
                  darkMode ? "bg-[#090D1C] border-[#1C2648]" : "bg-slate-50 border-slate-200"
                }`}>
                  <h4 className="text-xs font-semibold mb-3">Detected Objects</h4>
                  <div className="space-y-2.5">
                    {analysisResult.features.map((obj) => (
                      <div key={obj.id} className="flex items-center gap-2">
                        <span
                          className="w-3 h-3 rounded-sm shrink-0"
                          style={{ backgroundColor: obj.color }}
                        />
                        <span className={`text-[11px] truncate ${darkMode ? "text-slate-300" : "text-slate-700"}`}>
                          {obj.name}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Execution Trace Modal */}
      {showTraceModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className={`w-full max-w-2xl rounded-2xl border p-6 shadow-2xl ${
            darkMode ? "bg-[#0B1021] border-[#1F2B48]" : "bg-white border-slate-200"
          }`}>
            <div className="flex items-center justify-between pb-4 border-b border-inherit">
              <div className="flex items-center gap-2">
                <Terminal className="w-5 h-5 text-indigo-400" />
                <h3 className="font-bold text-sm">Auditable Execution Trace</h3>
              </div>
              <button
                onClick={() => setShowTraceModal(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <pre className={`mt-4 p-4 rounded-xl text-[11px] font-mono overflow-x-auto ${
              darkMode ? "bg-[#070A14] text-emerald-400" : "bg-slate-900 text-emerald-300"
            }`}>
              {JSON.stringify(analysisResult.executionTrace, null, 2)}
            </pre>
            <div className="mt-5 flex justify-end">
              <button
                onClick={() => setShowTraceModal(false)}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold"
              >
                Close Trace
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
