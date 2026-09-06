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
  Radio
} from "lucide-react";

export default function SatQueryApp() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState("Single Image");
  const [query, setQuery] = useState(
    "Identify the top 4 critical features and hazards. How many rivers are visible, and what percentage of the image is covered by urban settlement?"
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
    title: "Littoral Coastal Barrier & Urban Settlement Assessment",
    directQueryAnswers: {
      hydrology_and_waterways: "0 inland rivers detected. The western sector is open marine ocean water separated by a sand barrier berm.",
      urban_settlement_coverage: "Approximately 32% of the scene is covered by dense urban settlement, commercial structures, and road corridors.",
      hazards_and_vulnerabilities: "Vulnerability to coastal storm surge flooding, shoreline erosion, and close proximity of structures to the marine margin."
    },
    comprehensiveAssessment: "Multispectral satellite observation confirms a prominent littoral shoreline separating open marine waters from inland urban infrastructure.\n\nThe coastal margin is defined by an intertidal sand barrier that absorbs wave energy. Inland conurbation exhibits high building density transitioning into agricultural parcels.",
    confidenceScore: "0.95",
    previewUrl: "https://images.unsplash.com/photo-1524813686514-a57563d77d66?auto=format&fit=crop&w=1200&q=80",
    classDistribution: [
      { name: "Open Marine Waters", percentage: 38, color: "#0284C7", description: "Deep ocean surface showing strong NIR absorption." },
      { name: "Intertidal Sand Beach", percentage: 10, color: "#F59E0B", description: "Continuous coastal barrier sand berm." },
      { name: "Dense Urban Settlement", percentage: 32, color: "#E11D48", description: "High-density residential and commercial infrastructure." },
      { name: "Agricultural & Green Parcels", percentage: 20, color: "#10B981", description: "Structured crop parcels and vegetation canopy." }
    ],
    spectralMetrics: {
      "Water Index (NDWI)": "+0.58 (High Water Absorption)",
      "Built-Up Index (NDBI)": "+0.36 (Dense Impervious Surface)",
      "Canopy Vigor (NDVI)": "+0.44 (Cultivated Greenery)"
    },
    features: [
      { id: "f0", name: "Open Marine Waters", color: "#0284C7", percentage: 38, points: "20,50 340,50 380,500 320,950 20,950", center: [180, 500] },
      { id: "f1", name: "Intertidal Sand Beach", color: "#F59E0B", percentage: 10, points: "340,50 420,50 460,510 400,950 330,950", center: [390, 500] },
      { id: "f2", name: "Dense Urban Settlement", color: "#E11D48", percentage: 32, points: "430,520 720,510 740,880 410,880", center: [570, 700] },
      { id: "f3", name: "Agricultural Parcels", color: "#10B981", percentage: 20, points: "440,80 820,70 810,480 430,470", center: [630, 280] }
    ],
    executionTrace: {
      task: "single_image_vqa",
      detected_scene_category: "COASTAL",
      models_executed: [
        { name: "MultiSpectralContourPolygonEngine", params: { mode: "Single Image" } },
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
    "Use the optical and SAR images together to identify built-up and water."
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
    if (!image1) {
      alert("Please upload at least the primary satellite image first!");
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

  const isPairedMode = activeTab === "Change Detection" || activeTab === "Optical + SAR";

  return (
    <div className={`min-h-screen flex ${darkMode ? "bg-[#0B0F19] text-gray-100" : "bg-[#F8FAFC] text-slate-900"}`}>
      {/* Sidebar */}
      <aside
        className={`w-64 border-r flex flex-col justify-between p-5 shrink-0 ${
          darkMode ? "bg-[#111827] border-gray-800" : "bg-white border-slate-200 shadow-sm"
        }`}
      >
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
            <button
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-bold transition-colors ${
                darkMode ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/30" : "bg-indigo-50 text-indigo-700 border border-indigo-100"
              }`}
            >
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

        {/* Mode Toggle Button */}
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

      {/* Main Content Workspace */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        <div className="p-8 max-w-7xl w-full mx-auto space-y-6">
          {/* Top Query & Analysis Config Box */}
          <div
            className={`p-6 rounded-2xl border transition-all ${
              darkMode ? "bg-[#111827] border-gray-800 shadow-xl" : "bg-white border-slate-200 shadow-sm"
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <h2 className={`text-lg font-bold flex items-center gap-2 ${darkMode ? "text-white" : "text-slate-900"}`}>
                Interactive Geospatial Intelligence
                <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${darkMode ? "bg-indigo-950 text-indigo-300 border border-indigo-800" : "bg-indigo-50 text-indigo-700"}`}>
                  Polygon Grounding & VQA
                </span>
              </h2>

              {/* Mode Selection Tabs */}
              <div className="flex gap-1.5 p-1 rounded-xl bg-black/10 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                {[
                  { id: "Single Image", label: "Single Image" },
                  { id: "Change Detection", label: "Bi-Temporal Change" },
                  { id: "Optical + SAR", label: "Optical + SAR Pair" }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                      activeTab === tab.id
                        ? darkMode
                          ? "bg-indigo-600 text-white shadow"
                          : "bg-white text-indigo-700 shadow-sm"
                        : darkMode
                        ? "text-gray-400 hover:text-gray-200"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Bar */}
            <div
              className={`flex items-center rounded-xl border p-1.5 mb-3 transition-colors ${
                darkMode ? "bg-[#0B0F19] border-gray-700 focus-within:border-indigo-500" : "bg-slate-50 border-slate-300 focus-within:border-indigo-600"
              }`}
            >
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && executeAnalysis()}
                placeholder="Ask specific queries about rivers, hazards, urban footprint, or land-cover..."
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
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>

            {/* Suggested Hackathon Queries */}
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

            {/* Upload Area (Supports Single or Dual Paired Images) */}
            <div className={`pt-4 border-t ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
              <div className={`grid gap-4 ${isPairedMode ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
                {/* Primary Image Upload */}
                <div
                  onClick={() => fileInputRef1.current.click()}
                  className={`p-3.5 rounded-xl border-2 border-dashed flex items-center gap-3 cursor-pointer transition-all ${
                    darkMode
                      ? "border-gray-700 hover:border-indigo-500 bg-[#0B0F19]"
                      : "border-slate-300 hover:border-indigo-600 bg-slate-50"
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

                {/* Secondary Image Upload (Only shown in Paired Modes) */}
                {isPairedMode && (
                  <div
                    onClick={() => fileInputRef2.current.click()}
                    className={`p-3.5 rounded-xl border-2 border-dashed flex items-center gap-3 cursor-pointer transition-all ${
                      darkMode
                        ? "border-gray-700 hover:border-indigo-500 bg-[#0B0F19]"
                        : "border-slate-300 hover:border-indigo-600 bg-slate-50"
                    }`}
                  >
                    <input ref={fileInputRef2} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, true)} className="hidden" />
                    {activeTab === "Change Detection" ? (
                      <Clock className="w-5 h-5 text-amber-500 shrink-0" />
                    ) : (
                      <Radio className="w-5 h-5 text-emerald-500 shrink-0" />
                    )}
                    <div className="truncate">
                      <p className={`text-xs font-bold truncate ${darkMode ? "text-gray-200" : "text-slate-800"}`}>
                        {file2Name}
                      </p>
                      <p className={`text-[11px] ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                        {activeTab === "Change Detection" ? "Date 2 (Time 2) Image" : "Co-registered SAR Backscatter"}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* AI Response Card */}
          <div
            className={`p-6 rounded-2xl border transition-all ${
              darkMode ? "bg-[#111827] border-gray-800 shadow-xl" : "bg-white border-slate-200 shadow-sm"
            }`}
          >
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

            {/* Direct 3-Card Query Assessment */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {/* Hydrology */}
              <div
                className={`p-4 rounded-xl border ${
                  darkMode ? "bg-[#0E1726] border-sky-500/30" : "bg-sky-50/70 border-sky-200 shadow-sm"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Waves className="w-4 h-4 text-sky-500" />
                  <span className={`text-xs font-bold ${darkMode ? "text-sky-400" : "text-sky-900"}`}>
                    Hydrological & River Analysis
                  </span>
                </div>
                <p className={`text-xs leading-relaxed ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                  {analysisResult.directQueryAnswers.hydrology_and_waterways}
                </p>
              </div>

              {/* Urban */}
              <div
                className={`p-4 rounded-xl border ${
                  darkMode ? "bg-[#0E1726] border-rose-500/30" : "bg-rose-50/70 border-rose-200 shadow-sm"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Building2 className="w-4 h-4 text-rose-500" />
                  <span className={`text-xs font-bold ${darkMode ? "text-rose-400" : "text-rose-900"}`}>
                    Urban Settlement Coverage
                  </span>
                </div>
                <p className={`text-xs leading-relaxed ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                  {analysisResult.directQueryAnswers.urban_settlement_coverage}
                </p>
              </div>

              {/* Hazards */}
              <div
                className={`p-4 rounded-xl border ${
                  darkMode ? "bg-[#0E1726] border-amber-500/30" : "bg-amber-50/70 border-amber-200 shadow-sm"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-4 h-4 text-amber-500" />
                  <span className={`text-xs font-bold ${darkMode ? "text-amber-400" : "text-amber-900"}`}>
                    Hazards & Vulnerabilities
                  </span>
                </div>
                <p className={`text-xs leading-relaxed ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                  {analysisResult.directQueryAnswers.hazards_and_vulnerabilities}
                </p>
              </div>
            </div>

            {/* Multi-Paragraph Technical Report */}
            <div
              className={`p-4 rounded-xl border mb-6 ${
                darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-slate-50 border-slate-200"
              }`}
            >
              <div className="flex items-center gap-2 mb-2 text-xs font-bold uppercase tracking-wider text-indigo-500">
                <FileText className="w-3.5 h-3.5" /> Technical Intelligence Report
              </div>
              <p className={`text-xs leading-relaxed whitespace-pre-line ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                {analysisResult.comprehensiveAssessment}
              </p>
            </div>

            {/* Spectral Metrics Grid */}
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

            {/* Interactive Visualizer + 4 Classes */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Image Viewport with Polygon Vector Overlays */}
              <div
                className={`lg:col-span-8 rounded-2xl overflow-hidden border relative bg-black ${
                  darkMode ? "border-gray-800" : "border-slate-200"
                }`}
              >
                {/* Floating Tools */}
                <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-black/75 backdrop-blur-md p-1.5 rounded-xl border border-white/10 shadow-lg">
                  <button
                    onClick={() => setZoomLevel((z) => Math.min(z + 0.2, 2.2))}
                    className="p-1.5 text-gray-300 hover:text-white"
                    title="Zoom In"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setZoomLevel((z) => Math.max(z - 0.2, 0.8))}
                    className="p-1.5 text-gray-300 hover:text-white"
                    title="Zoom Out"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setShowOverlays(!showOverlays)}
                    className={`p-1.5 rounded-lg transition-colors ${
                      showOverlays ? "text-indigo-400 bg-indigo-500/20" : "text-gray-400"
                    }`}
                    title="Toggle Polygons"
                  >
                    <Layers className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div
                  className="relative w-full h-[450px] overflow-hidden"
                  style={{ transform: `scale(${zoomLevel})`, transformOrigin: "center center" }}
                >
                  <img
                    src={analysisResult.previewUrl}
                    alt="Satellite Scene"
                    className="w-full h-full object-cover select-none"
                  />

                  {/* SVG Vector Layer rendering True Organic Polygons */}
                  {showOverlays && (
                    <svg
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      viewBox="0 0 1024 1024"
                      preserveAspectRatio="none"
                    >
                      {analysisResult.features.map((f, idx) => {
                        const isHovered = hoveredIdx === idx;
                        return (
                          <g key={f.id} className="transition-all duration-300">
                            {/* Organic Polygon */}
                            <polygon
                              points={f.points}
                              fill={isHovered ? `${f.color}77` : `${f.color}33`}
                              stroke={f.color}
                              strokeWidth={isHovered ? "4" : "2.5"}
                              strokeDasharray={idx === 1 ? "6,4" : "none"}
                            />
                            {/* Centroid Labeled Marker Pin */}
                            <circle
                              cx={f.center[0]}
                              cy={f.center[1]}
                              r={isHovered ? "18" : "15"}
                              fill={f.color}
                              stroke="#FFFFFF"
                              strokeWidth="2.5"
                              className="filter drop-shadow"
                            />
                            <text
                              x={f.center[0]}
                              y={f.center[1] + 4.5}
                              textAnchor="middle"
                              fill="#FFFFFF"
                              fontSize="12"
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

              {/* 4 Detected Class Distribution Cards */}
              <div className="lg:col-span-4 space-y-3">
                <div className={`flex items-center justify-between pb-2 border-b ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
                  <h4 className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                    <BarChart3 className="w-3.5 h-3.5 text-indigo-500" /> Grounded Land Distribution
                  </h4>
                  <span className={`text-[10px] font-mono font-bold ${darkMode ? "text-gray-500" : "text-slate-400"}`}>
                    4 Classes
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

                      {/* Percentage Bar */}
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

      {/* Auditable Execution Trace Modal (for Hackathon Judges) */}
      {showTraceModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div
            className={`border p-6 rounded-2xl max-w-xl w-full shadow-2xl ${
              darkMode ? "bg-[#111827] border-gray-700 text-white" : "bg-white border-slate-200 text-slate-900"
            }`}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-sm text-indigo-500 flex items-center gap-2">
                <Terminal className="w-4 h-4" /> Auditable Agentic Trace
              </h3>
              <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-500 font-mono">
                Judge Review Ready
              </span>
            </div>
            <p className={`text-xs mb-3 ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
              Observable execution summary verifying input compatibility, task routing, and model orchestration:
            </p>
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
