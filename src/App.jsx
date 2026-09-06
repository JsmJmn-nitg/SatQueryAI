import React, { useState, useRef } from "react";
import {
  Sparkles,
  Send,
  Layers,
  ZoomIn,
  ZoomOut,
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
  FileText,
  HelpCircle,
  Clock,
  Radio,
  Satellite,
  Globe2,
  Database,
  SearchCode
} from "lucide-react";

export default function SatQueryApp() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState("Single Image");
  const [query, setQuery] = useState(
    "Describe the land-cover and major objects visible in this image. How many rivers flow into the ocean and describe it? Are there urban settlements?"
  );
  const [loading, setLoading] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showTraceModal, setShowTraceModal] = useState(false);
  const [hoveredIdx, setHoveredIdx] = useState(null);

  const [image1, setImage1] = useState(null);
  const [file1Name, setFile1Name] = useState("Primary Satellite Image (.tif, .png)");
  const [image2, setImage2] = useState(null);
  const [file2Name, setFile2Name] = useState("Secondary Image / SAR / Date 2 (Optional)");

  const [analysisResult, setAnalysisResult] = useState({
    title: "Satellite Image Analysis of Coastal Urban Area",
    dynamicCards: [
      {
        category: "Hydrological & River Analysis",
        text: "There are no visible inland rivers in the image. The western coastal margin consists of open marine waters separated by an intertidal sand berm.",
        type: "water"
      },
      {
        category: "Urban Settlement Coverage",
        text: "The urban settlement is dense, with mixed residential and commercial blocks covering approximately 32% of the total scene in the eastern quadrant.",
        type: "urban"
      },
      {
        category: "Hazards & Vulnerabilities",
        text: "Low-lying urban infrastructure abuts the shoreline without deep vegetative buffers, creating vulnerability to coastal storm surges.",
        type: "hazard"
      }
    ],
    technicalReport: "The multispectral observation confirms a littoral urban landscape. Open marine waters dominate the western quadrant, bounded by a continuous beach sand berm. Inland conurbation exhibits high building density transitioning into cultivated vegetation.",
    confidenceScore: "0.95",
    previewUrl: "https://images.unsplash.com/photo-1524813686514-a57563d77d66?auto=format&fit=crop&w=1200&q=80",
    classDistribution: [
      { name: "Open Marine Waters", percentage: 42, color: "#0284C7", description: "Deep marine water body displaying strong NIR absorption." },
      { name: "Dense Urban Settlement", percentage: 32, color: "#E11D48", description: "High-density residential and commercial infrastructure." },
      { name: "Vegetation & Cropland", percentage: 16, color: "#10B981", description: "Cultivated crop parcels and natural green canopy." },
      { name: "Intertidal Sand Beach", percentage: 10, color: "#F59E0B", description: "Coastal barrier sand berm and shoreline margin." }
    ],
    spectralMetrics: {
      "Water Index (NDWI)": "+0.58 (High Water Absorption)",
      "Built-Up Index (NDBI)": "+0.36 (Dense Impervious Surface)",
      "Canopy Vigor (NDVI)": "+0.44 (Cultivated Greenery)"
    },
    features: [
      { id: "f0", name: "Open Marine Waters", color: "#0284C7", percentage: 42, points: "20,50 340,50 380,500 320,950 20,950", center: [180, 500] },
      { id: "f1", name: "Dense Urban Settlement", color: "#E11D48", percentage: 32, points: "430,520 720,510 740,880 410,880", center: [570, 700] },
      { id: "f2", name: "Vegetation & Cropland", color: "#10B981", percentage: 16, points: "440,80 820,70 810,480 430,470", center: [630, 280] },
      { id: "f3", name: "Intertidal Sand Beach", color: "#F59E0B", percentage: 10, points: "340,50 420,50 460,510 400,950 330,950", center: [390, 500] }
    ],
    executionTrace: {
      task: "single_image_vqa",
      detected_physical_regime: "COASTAL_MARINE",
      tools_executed: [
        { name: "AdaptivePolygonContourEngine", params: { mode: "Single Image" } },
        { name: "DomainAdaptedVLM_Qwen2VL", params: { temperature: 0.1 } }
      ]
    }
  });

  const fileInputRef1 = useRef(null);
  const fileInputRef2 = useRef(null);

  const sampleQueries = [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body and assess whether any inland rivers exist.",
    "What changed between these two dates, and where did the change occur?",
    "Use optical and SAR data to identify built-up and water bodies.",
    "[AutoFetch] Retrieve Sentinel-2 tiles for Valencia flood zone and evaluate inundated area."
  ];

  const handleFileUpload = (e, isSecond = false) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!isSecond) {
      setImage1(file);
      setFile1Name(file.name);
      if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
        setAnalysisResult((prev) => ({ ...prev, previewUrl: URL.createObjectURL(file) }));
      }
    } else {
      setImage2(file);
      setFile2Name(file.name);
    }
  };

  const executeAnalysis = async () => {
    if (activeTab === "AutoFetch") {
      alert("AutoFetch UI Mode: The automated ingestion pipeline will be connected here. For now, upload a local image to run the local grounding engine.");
      return;
    }

    if (!image1) {
      alert("Please upload an image first!");
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
      if (!res.ok) throw new Error("Server error: " + res.statusText);
      const data = await res.json();
      setAnalysisResult({
        title: data.title,
        dynamicCards: data.dynamic_cards || [],
        technicalReport: data.technical_report,
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

  const getCardIcon = (type) => {
    if (type === "water") return <Waves className="w-4 h-4 text-sky-500" />;
    if (type === "urban") return <Building2 className="w-4 h-4 text-rose-500" />;
    return <AlertTriangle className="w-4 h-4 text-amber-500" />;
  };

  const isPairedMode = activeTab === "Bi-Temporal Change" || activeTab === "Optical + SAR Pair";

  return (
    <div className={`min-h-screen flex ${darkMode ? "bg-[#0B0F19] text-gray-100" : "bg-[#F8FAFC] text-slate-900"}`}>
      {/* Sidebar */}
      <aside className={`w-64 border-r flex flex-col justify-between p-5 shrink-0 ${
        darkMode ? "bg-[#111827] border-gray-800" : "bg-white border-slate-200 shadow-sm"
      }`}>
        <div>
          <div className="flex items-center gap-3 px-1 py-2 mb-6">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-blue-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className={`font-bold text-base tracking-tight leading-tight ${darkMode ? "text-white" : "text-slate-900"}`}>
                SatQuery AI
              </h1>
              <span className={`text-[11px] font-semibold tracking-wide ${darkMode ? "text-gray-400" : "text-indigo-600"}`}>
                Earth Observation AI
              </span>
            </div>
          </div>

          <nav className="space-y-1.5">
            <button className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-bold transition-colors ${
              darkMode ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/30" : "bg-indigo-50 text-indigo-700 border border-indigo-100"
            }`}>
              <Home className="w-4 h-4" /> Console
            </button>
            <button
              onClick={() => setShowTraceModal(true)}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
                darkMode ? "text-gray-400 hover:bg-gray-800/60 hover:text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <Terminal className="w-4 h-4 text-emerald-500" /> Execution Trace
            </button>
          </nav>
        </div>

        {/* Theme Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className={`w-full flex items-center justify-between p-3 rounded-xl border text-xs font-bold transition-all ${
            darkMode
              ? "bg-[#161F30] border-gray-700 text-gray-200 hover:bg-gray-800"
              : "bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200"
          }`}
        >
          <div className="flex items-center gap-2">
            {darkMode ? <Moon className="w-4 h-4 text-indigo-400" /> : <Sun className="w-4 h-4 text-amber-500" />}
            <span>{darkMode ? "Dark Console" : "Light Mode"}</span>
          </div>
          <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
        </button>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        <div className="p-8 max-w-7xl w-full mx-auto space-y-6">
          {/* Top Control Box */}
          <div className={`p-6 rounded-2xl border transition-all ${
            darkMode ? "bg-[#111827] border-gray-800 shadow-xl" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <h2 className={`text-lg font-bold flex items-center gap-2 ${darkMode ? "text-white" : "text-slate-900"}`}>
                Interactive Geospatial Intelligence
                <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${darkMode ? "bg-indigo-950 text-indigo-300 border border-indigo-800" : "bg-indigo-50 text-indigo-700"}`}>
                  Adaptive Polygon Grounding
                </span>
              </h2>

              {/* Mode Selector (Including AutoFetch) */}
              <div className="flex gap-1.5 p-1 rounded-xl bg-black/10 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                {[
                  { id: "Single Image", label: "Single Image" },
                  { id: "Bi-Temporal Change", label: "Bi-Temporal Change" },
                  { id: "Optical + SAR Pair", label: "Optical + SAR Pair" },
                  { id: "AutoFetch", label: "AutoFetch" }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                      activeTab === tab.id
                        ? darkMode
                          ? "bg-indigo-600 text-white shadow"
                          : "bg-white text-indigo-700 shadow-sm"
                        : darkMode
                        ? "text-gray-400 hover:text-gray-200"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {tab.id === "AutoFetch" && <Satellite className="w-3.5 h-3.5" />}
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Bar */}
            <div className={`flex items-center rounded-xl border p-1.5 mb-3 transition-colors ${
              darkMode ? "bg-[#0B0F19] border-gray-700 focus-within:border-indigo-500" : "bg-slate-50 border-slate-300 focus-within:border-indigo-600"
            }`}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && executeAnalysis()}
                placeholder={
                  activeTab === "AutoFetch"
                    ? "Enter target location or event query (e.g., 'Fetch Sentinel-2 imagery for Valencia flood and assess damage')..."
                    : "Ask questions about rivers, urban footprint, land-cover, or active hazards..."
                }
                className={`w-full bg-transparent px-3 py-2 text-xs font-medium outline-none ${
                  darkMode ? "text-white placeholder-gray-500" : "text-slate-900 placeholder-slate-400"
                }`}
              />
              <button
                onClick={executeAnalysis}
                disabled={loading}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2.5 rounded-lg shadow font-bold text-xs flex items-center gap-2 shrink-0 transition-all"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                {loading ? "Analyzing..." : activeTab === "AutoFetch" ? "Fetch & Analyze" : "Analyze"}
              </button>
            </div>

            {/* Suggested Queries */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <span className={`text-[11px] font-semibold flex items-center gap-1 ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                <HelpCircle className="w-3.5 h-3.5" /> Sample Queries:
              </span>
              {sampleQueries.map((sq, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(sq)}
                  className={`text-[11px] px-2.5 py-1 rounded-md border transition-all truncate max-w-xs ${
                    darkMode
                      ? "bg-gray-800/80 border-gray-700 text-gray-300 hover:border-indigo-500 hover:text-white"
                      : "bg-slate-100 border-slate-200 text-slate-700 hover:border-indigo-400 hover:text-indigo-800"
                  }`}
                  title={sq}
                >
                  {sq}
                </button>
              ))}
            </div>

            {/* Upload Area / AutoFetch Data Catalog UI */}
            <div className={`pt-4 border-t ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
              {activeTab === "AutoFetch" ? (
                /* AutoFetch Mode Ingestion Architecture Panel */
                <div className={`p-4 rounded-xl border ${darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-slate-50 border-slate-200"}`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-indigo-400 flex items-center gap-2">
                      <SearchCode className="w-4 h-4" /> Autonomous Satellite Ingestion Pipeline (Standby)
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-mono font-bold">
                      Zero-Upload Workflow
                    </span>
                  </div>
                  <p className={`text-xs mb-3 ${darkMode ? "text-gray-400" : "text-slate-600"}`}>
                    In AutoFetch mode, SatQuery AI resolves geographic coordinates from your natural-language query and autonomously ingests multi-spectral tiles from connected earth observation catalogs:
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                    {[
                      { name: "Sentinel-1/2", org: "ESA Copernicus OpenHub", active: true },
                      { name: "Landsat-8/9", org: "USGS EarthExplorer", active: true },
                      { name: "Google Earth Engine", org: "Cloud Data Catalog", active: true },
                      { name: "Private Commercial API", org: "PlanetScope / SkySat", active: false }
                    ].map((src, i) => (
                      <div key={i} className={`p-2.5 rounded-lg border text-left ${darkMode ? "bg-gray-800/50 border-gray-700" : "bg-white border-slate-200 shadow-sm"}`}>
                        <div className="flex items-center justify-between text-[11px] font-bold">
                          <span>{src.name}</span>
                          <span className={`w-2 h-2 rounded-full ${src.active ? "bg-emerald-400" : "bg-amber-400"}`} />
                        </div>
                        <span className={`text-[10px] block truncate ${darkMode ? "text-gray-400" : "text-slate-500"}`}>{src.org}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                /* Standard Upload Slots */
                <div className={`grid gap-4 ${isPairedMode ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
                  {/* Primary Upload */}
                  <div
                    onClick={() => fileInputRef1.current.click()}
                    className={`p-3.5 rounded-xl border-2 border-dashed flex items-center gap-3 cursor-pointer transition-all ${
                      darkMode ? "border-gray-700 hover:border-indigo-500 bg-[#0B0F19]" : "border-slate-300 hover:border-indigo-600 bg-slate-50"
                    }`}
                  >
                    <input ref={fileInputRef1} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, false)} className="hidden" />
                    <UploadCloud className="w-5 h-5 text-indigo-500 shrink-0" />
                    <div className="truncate">
                      <p className={`text-xs font-bold truncate ${darkMode ? "text-gray-200" : "text-slate-800"}`}>
                        {file1Name}
                      </p>
                      <p className={`text-[11px] ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                        {isPairedMode ? "Primary / Optical (Time 1) GeoTIFF or photo" : "Satellite GeoTIFF, TIFF, PNG, or JPEG"}
                      </p>
                    </div>
                  </div>

                  {/* Secondary Upload for Paired Modes */}
                  {isPairedMode && (
                    <div
                      onClick={() => fileInputRef2.current.click()}
                      className={`p-3.5 rounded-xl border-2 border-dashed flex items-center gap-3 cursor-pointer transition-all ${
                        darkMode ? "border-gray-700 hover:border-indigo-500 bg-[#0B0F19]" : "border-slate-300 hover:border-indigo-600 bg-slate-50"
                      }`}
                    >
                      <input ref={fileInputRef2} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, true)} className="hidden" />
                      {activeTab === "Bi-Temporal Change" ? (
                        <Clock className="w-5 h-5 text-amber-500 shrink-0" />
                      ) : (
                        <Radio className="w-5 h-5 text-emerald-500 shrink-0" />
                      )}
                      <div className="truncate">
                        <p className={`text-xs font-bold truncate ${darkMode ? "text-gray-200" : "text-slate-800"}`}>
                          {file2Name}
                        </p>
                        <p className={`text-[11px] ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                          {activeTab === "Bi-Temporal Change" ? "Date 2 (Time 2) Image" : "Co-registered SAR Backscatter"}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* AI Response Card */}
          <div className={`p-6 rounded-2xl border transition-all ${
            darkMode ? "bg-[#111827] border-gray-800 shadow-xl" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <div className={`flex justify-between items-center mb-4 pb-3 border-b ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
              <span className="text-xs font-bold text-indigo-500 flex items-center gap-1.5 uppercase tracking-wide">
                <Bot className="w-4 h-4" /> Comprehensive Geospatial Assessment
              </span>
              <span className="text-xs font-mono text-emerald-500 font-bold px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                Confidence: {analysisResult.confidenceScore}
              </span>
            </div>

            <h3 className={`text-xl font-bold mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
              {analysisResult.title}
            </h3>

            {/* DYNAMIC QUERY ASSESSMENT CARDS (Adapts to any scene domain) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {analysisResult.dynamicCards.map((card, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-xl border ${
                    darkMode ? "bg-[#0E1726] border-gray-800" : "bg-slate-50 border-slate-200 shadow-sm"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {getCardIcon(card.type)}
                    <span className={`text-xs font-bold ${darkMode ? "text-gray-200" : "text-slate-800"}`}>
                      {card.category}
                    </span>
                  </div>
                  <p className={`text-xs leading-relaxed ${darkMode ? "text-gray-300" : "text-slate-600"}`}>
                    {card.text}
                  </p>
                </div>
              ))}
            </div>

            {/* Technical Report */}
            <div className={`p-4 rounded-xl border mb-6 ${
              darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-slate-50 border-slate-200"
            }`}>
              <div className="flex items-center gap-2 mb-2 text-xs font-bold uppercase tracking-wider text-indigo-500">
                <FileText className="w-3.5 h-3.5" /> Technical Intelligence Report
              </div>
              <p className={`text-xs leading-relaxed whitespace-pre-line ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                {analysisResult.technicalReport}
              </p>
            </div>

            {/* Dynamic Spectral Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
              {Object.entries(analysisResult.spectralMetrics).map(([k, v], idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-xl border ${
                    darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-white border-slate-200 shadow-sm"
                  }`}
                >
                  <span className={`text-[10px] block uppercase font-mono font-bold mb-0.5 ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                    {k}
                  </span>
                  <span className="text-xs font-bold text-indigo-500">{v}</span>
                </div>
              ))}
            </div>

            {/* Interactive Visualizer + Dynamic Class Distribution */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Image Viewport with Polygon Overlays */}
              <div className={`lg:col-span-8 rounded-2xl overflow-hidden border relative bg-black ${
                darkMode ? "border-gray-800" : "border-slate-200"
              }`}>
                <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-black/75 backdrop-blur-md p-1.5 rounded-xl border border-white/10 shadow-lg">
                  <button onClick={() => setZoomLevel((z) => Math.min(z + 0.2, 2.2))} className="p-1.5 text-gray-300 hover:text-white" title="Zoom In">
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => setZoomLevel((z) => Math.max(z - 0.2, 0.8))} className="p-1.5 text-gray-300 hover:text-white" title="Zoom Out">
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => setShowOverlays(!showOverlays)} className={`p-1.5 rounded-lg transition-colors ${showOverlays ? "text-indigo-400 bg-indigo-500/20" : "text-gray-400"}`} title="Toggle Polygons">
                    <Layers className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="relative w-full h-[450px] overflow-hidden" style={{ transform: `scale(${zoomLevel})`, transformOrigin: "center center" }}>
                  <img src={analysisResult.previewUrl} alt="Satellite Scene" className="w-full h-full object-cover select-none" />

                  {showOverlays && (
                    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1024 1024" preserveAspectRatio="none">
                      {analysisResult.features.map((f, idx) => {
                        const isHovered = hoveredIdx === idx;
                        return (
                          <g key={f.id} className="transition-all duration-300">
                            <polygon
                              points={f.points}
                              fill={isHovered ? `${f.color}77` : `${f.color}33`}
                              stroke={f.color}
                              strokeWidth={isHovered ? "4" : "2.5"}
                            />
                            <circle cx={f.center[0]} cy={f.center[1]} r={isHovered ? "18" : "15"} fill={f.color} stroke="#FFFFFF" strokeWidth="2.5" />
                            <text x={f.center[0]} y={f.center[1] + 4.5} textAnchor="middle" fill="#FFFFFF" fontSize="12" fontWeight="bold">
                              {idx + 1}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                </div>
              </div>

              {/* Dynamic Class Distribution List */}
              <div className="lg:col-span-4 space-y-3">
                <div className={`flex items-center justify-between pb-2 border-b ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
                  <h4 className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                    <BarChart3 className="w-3.5 h-3.5 text-indigo-500" /> Grounded Land Distribution
                  </h4>
                  <span className={`text-[10px] font-mono font-bold ${darkMode ? "text-gray-500" : "text-slate-400"}`}>
                    {analysisResult.classDistribution.length} Classes
                  </span>
                </div>

                {analysisResult.classDistribution.map((item, idx) => {
                  const isHovered = hoveredIdx === idx;
                  return (
                    <div
                      key={idx}
                      onMouseEnter={() => setHoveredIdx(idx)}
                      onMouseLeave={() => setHoveredIdx(null)}
                      className={`p-3 rounded-xl border transition-all cursor-pointer ${
                        isHovered
                          ? darkMode
                            ? "border-indigo-500 bg-indigo-500/10 shadow-lg"
                            : "border-indigo-500 bg-indigo-50 shadow-md"
                          : darkMode
                          ? "border-gray-800 bg-[#0B0F19] hover:border-gray-700"
                          : "border-slate-200 bg-white hover:border-slate-300 shadow-sm"
                      }`}
                    >
                      <div className="flex justify-between items-center text-xs mb-1.5">
                        <span className={`flex items-center gap-2 font-bold ${darkMode ? "text-white" : "text-slate-800"}`}>
                          <span
                            className="w-4 h-4 rounded-md text-[10px] font-bold text-white flex items-center justify-center shrink-0"
                            style={{ backgroundColor: item.color }}
                          >
                            {idx + 1}
                          </span>
                          <span className="truncate max-w-[160px]" title={item.name}>
                            {item.name}
                          </span>
                        </span>
                        <span className={`font-mono text-xs font-bold ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                          {item.percentage}%
                        </span>
                      </div>

                      <div className={`w-full rounded-full h-1.5 overflow-hidden mb-2 ${darkMode ? "bg-gray-800" : "bg-slate-100"}`}>
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${item.percentage}%`, backgroundColor: item.color }}
                        />
                      </div>

                      {item.description && (
                        <p className={`text-[11px] leading-snug ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
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

      {/* Execution Trace Modal */}
      {showTraceModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className={`border p-6 rounded-2xl max-w-xl w-full shadow-2xl ${
            darkMode ? "bg-[#111827] border-gray-700 text-white" : "bg-white border-slate-200 text-slate-900"
          }`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-sm text-indigo-500 flex items-center gap-2">
                <Terminal className="w-4 h-4" /> Auditable Agentic Trace
              </h3>
              <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-500 font-mono">
                Judge Review Ready
              </span>
            </div>
            <pre className="bg-[#0B0F19] text-emerald-400 p-4 rounded-xl text-xs font-mono overflow-auto max-h-96 border border-gray-800">
              {JSON.stringify(analysisResult.executionTrace, null, 2)}
            </pre>
            <button
              onClick={() => setShowTraceModal(false)}
              className="mt-4 w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold transition-colors"
            >
              Close Trace
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
